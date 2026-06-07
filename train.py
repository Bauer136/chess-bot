"""Train the per-square piece classifier on labeled crops in data/.

Two-phase MobileNetV2 transfer learning (matches docs/training_plan.md):
    phase 1: backbone frozen, train the head
    phase 2: top N layers unfrozen, fine-tune at 10x lower LR

The dataset on disk lives in `data/<class>/*.jpg` where class is one of:
    empty P N B R Q K p n b r q k
`empty` maps to the `.` label the rest of the pipeline expects. Class names
are passed explicitly so the saved model's output index order is stable
across machines (alphabetical default would order them B,K,N,P,Q,R,b,empty,...
and decouple from INDEX_TO_LABEL in classifier.py).

Usage:
    python train.py                       # defaults: 96x96, 32 batch, 10+20 epochs
    python train.py --input-size 64
    python train.py --epochs-head 5 --epochs-finetune 0
    # Warm-start from a synthetic-pretrained checkpoint:
    python train.py --data-dir data --init-from models/classifier_synth.keras \
        --epochs-head 0 --epochs-finetune 15
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "classifier.keras"
DEFAULT_EVAL = PROJECT_ROOT / "models" / "eval.json"

# Order matches classifier.INDEX_TO_LABEL. `empty` -> `.` mapping lives in
# the inference side; here we just keep the index order consistent.
CLASS_NAMES = ["empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
NUM_CLASSES = len(CLASS_NAMES)
SEED = 42
VAL_SPLIT = 0.15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--eval-output", default=str(DEFAULT_EVAL))
    p.add_argument("--input-size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs-head", type=int, default=10)
    p.add_argument("--epochs-finetune", type=int, default=20)
    p.add_argument("--finetune-layers", type=int, default=50,
                   help="Number of top MobileNetV2 layers to unfreeze in phase 2.")
    p.add_argument(
        "--init-from", default=None,
        help=(
            "Path to a saved .keras model to start from instead of fresh "
            "imagenet weights. Use this to fine-tune a synthetic-pretrained "
            "model on real data. When set, --input-size is overridden to the "
            "loaded model's expected input. Pair with --epochs-head 0 so the "
            "head's existing weights aren't retrained from scratch."
        ),
    )
    return p.parse_args()


def load_datasets(data_dir: Path, input_size: int, batch_size: int):
    common = dict(
        directory=str(data_dir),
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=(input_size, input_size),
        batch_size=batch_size,
        seed=SEED,
        validation_split=VAL_SPLIT,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", **common)
    return train_ds, val_ds


def compute_class_weights(data_dir: Path) -> dict:
    """Weight loss inversely with class frequency.

    Real games skew heavily empty (~50%) and pawns (~20%); kings/queens are
    ~1/64. Sample-counting once up front is cheaper than oversampling on
    the data pipeline and gives equivalent gradients."""
    counts = []
    for cls in CLASS_NAMES:
        folder = data_dir / cls
        counts.append(sum(1 for _ in folder.glob("*.jpg")) if folder.exists() else 0)
    counts = np.array(counts, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        raise SystemExit(f"no .jpg crops found under {data_dir}")
    weights = {}
    for i, n in enumerate(counts):
        weights[i] = float(total / (NUM_CLASSES * n)) if n > 0 else 1.0
    print("[train] class counts:", dict(zip(CLASS_NAMES, counts.astype(int).tolist())))
    return weights


def build_augmentation() -> tf.keras.Sequential:
    # No flips: pieces have a top/bottom orientation, and a flip would also
    # swap each square's diagonal color, misrepresenting bishop identity.
    return tf.keras.Sequential([
        layers.RandomRotation(10 / 360),
        layers.RandomBrightness(0.20),
        layers.RandomContrast(0.15),
        layers.RandomZoom(0.10),
    ], name="augment")


def build_model(input_size: int):
    inputs = tf.keras.Input(shape=(input_size, input_size, 3))
    x = build_augmentation()(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(input_size, input_size, 3),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), backbone


def find_backbone(model) -> tf.keras.Model:
    """Locate the MobileNetV2 sub-model inside a loaded checkpoint so phase 2
    can selectively unfreeze its top layers. MobileNetV2's auto-generated
    name starts with `mobilenetv2_`, which is stable across Keras versions."""
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            return layer
    raise SystemExit(
        "could not find MobileNetV2 backbone in --init-from model; "
        f"layers were: {[l.name for l in model.layers]}"
    )


def evaluate(model, val_ds) -> dict:
    """Per-class accuracy + confusion matrix on the validation split."""
    y_true, y_pred = [], []
    for xb, yb in val_ds:
        probs = model.predict(xb, verbose=0)
        y_true.append(yb.numpy())
        y_pred.append(probs.argmax(axis=1))
    if not y_true:
        return {"top1_accuracy": None, "per_class": {}, "confusion_matrix": [], "class_names": CLASS_NAMES}
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        per_class[cls] = {
            "support": total,
            "correct": correct,
            "accuracy": (correct / total) if total else None,
        }
    return {
        "top1_accuracy": float((y_true == y_pred).mean()),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output = Path(args.output).resolve()
    eval_output = Path(args.eval_output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load or build the model before datasets so --init-from can pin the
    # input size to whatever the saved checkpoint expects.
    if args.init_from:
        print(f"[train] loading initial weights from {args.init_from}")
        model = tf.keras.models.load_model(args.init_from)
        backbone = find_backbone(model)
        model_size = model.input_shape[1]
        if model_size != args.input_size:
            print(f"[train] --init-from model uses {model_size}x{model_size}; "
                  f"overriding --input-size {args.input_size}")
            args.input_size = model_size
    else:
        model, backbone = build_model(args.input_size)

    train_ds, val_ds = load_datasets(data_dir, args.input_size, args.batch_size)
    class_weights = compute_class_weights(data_dir)

    # Cache + prefetch — disk reads dominate on CPU training.
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1024, seed=SEED).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    if args.epochs_head > 0:
        model.compile(
            optimizer=optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs_head,
            class_weight=class_weights,
        )

    if args.epochs_finetune > 0:
        backbone.trainable = True
        for layer in backbone.layers[:-args.finetune_layers]:
            layer.trainable = False
        model.compile(
            optimizer=optimizers.Adam(1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs_finetune,
            class_weight=class_weights,
            callbacks=[ckpt],
        )

    # Reload the best checkpoint if one was saved; otherwise save the
    # current weights so eval and inference see the same model.
    if output.exists():
        best = tf.keras.models.load_model(output)
    else:
        best = model
        best.save(output)

    metrics = evaluate(best, val_ds)
    eval_output.write_text(json.dumps(metrics, indent=2))
    print(f"[train] saved model to {output}")
    print(f"[train] saved eval to {eval_output}")
    print(f"[train] top-1 accuracy: {metrics['top1_accuracy']}")


if __name__ == "__main__":
    main()
