# CNN piece classifier — training plan

This is the plan for replacing the stub in `classifier.py` with a real
trained model. Nothing here is implemented yet; this is a thinking
document.

## Goal

A function `predict(squares_batch: np.ndarray) -> List[str]` that takes a
`(64, H, W, 3)` uint8 batch in a8 → h1 order and returns 64 labels, one
per square, where each label is one of:

```
P N B R Q K  (white)
p n b r q k  (black)
.            (empty)
```

This matches the existing stub's signature exactly. The rest of the
pipeline (Stage 7 board reconstruction, Stage 8 Stockfish) does not need
to change.

Target accuracy: ≥ 95% per-square top-1, ≥ 99% on `empty`. Per-square
errors are not equally costly — a single wrong square breaks board
reconstruction in Stage 7. With 95% per-square that's roughly
`1 - 0.95**64 = 96%` boards wrong, which is too high. We probably need
≥ 99.5% per-square, and we accept that won't happen out of the gate.

## Dataset

### Label set
13 classes:
```
0=. 1=P 2=N 3=B 4=R 5=Q 6=K 7=p 8=n 9=b 10=r 11=q 12=k
```

Folder layout (on-disk dataset):
```
data/
  empty/    *.jpg
  P/        *.jpg
  N/        *.jpg
  ... etc, 13 folders total
```
`.` is a reserved character in some filesystems, so we use the folder
name `empty` and map it to `.` in code.

### Size targets
- v0 (working CNN at all): 500 crops/class minimum, ~6.5k total.
- v1 (production-quality): 2-5k crops/class, ~25-65k total.
- Class balance: weight the loss instead of oversampling, because the
  natural distribution from real games skews heavily toward `empty`
  (50%+) and pawns (20%+). King/queen are 1/64 of any position.

### Acquisition strategy (hybrid)
1. **Synthetic pretrain.** Render N board positions with
   `python-chess` + piece glyphs over varied wood/marble textures and
   lighting, slice each into 64 labeled crops via the existing
   `segment_board`. Pros: thousands of perfectly-labeled crops in
   minutes. Cons: 2D glyphs don't match 3D pieces, so the model trained
   on synthetic alone will struggle with the real camera feed.
2. **Self-captured fine-tune.** Use `python main.py --save-squares` on
   real games or hand-arranged positions; manually drag crops into the
   `data/{class}/` folders. Target 30-50 captured positions, which
   yields ~2000-3200 crops. Most go to `empty`; pieces are ~30%.

If we only have time for one path, **self-captured wins for actual
deployment** because it's domain-matched. Synthetic is only worth it as
a warm-start. Decision deferred until we see how painful the labeling is.

### Augmentation
Applied at training time, not on disk:
- Random rotation ±10° (board may not be perfectly aligned even after
  rectification).
- Random brightness ±20%, contrast ±15% (lighting changes during a game).
- Random crop & resize back to 64×64 (jitter the square boundaries).
- **No horizontal or vertical flip** — pieces have orientation
  (top-of-board kings face the opposite way from bottom-of-board
  kings), and flipping would also swap the bishop's square-color
  identity in a misleading way.

## Model

Transfer learning, per `notes.md`. Default choice: MobileNetV2.

- Input: 64×64×3 RGB float32 in [0, 1].
- Backbone: `tf.keras.applications.MobileNetV2(include_top=False,
  weights='imagenet', input_shape=(64,64,3))`. MobileNetV2's smallest
  recommended input is 96; at 64 some layers degrade. If accuracy
  suffers, bump to 96×96 (cheap — segmentation is once-per-move, not
  per-frame).
- Head: GlobalAveragePooling2D → Dropout(0.3) → Dense(13, softmax).
- Phase 1 training: freeze backbone, train head only, ~10 epochs.
- Phase 2 training: unfreeze top N layers (50 or so), fine-tune with
  10× lower LR, ~20 epochs.

Saved as `models/classifier.keras` (Keras 3 native format).

## Training script

New file `train.py`, run standalone. Not wired into `main.py`.

Sketch:
```
load dataset from data/ via tf.keras.utils.image_dataset_from_directory
  - image_size=(64,64) or (96,96)
  - validation_split=0.15
  - shuffle=True, seed=42
apply augmentation pipeline (Sequential of RandomRotation etc.)
build model (phase 1)
compile with sparse_categorical_crossentropy, Adam(1e-3),
  class_weight=auto
fit, save best by val_loss
unfreeze top 50 layers
recompile with Adam(1e-4)
fit more
save final to models/classifier.keras
```

CLI flags: `--data-dir`, `--epochs-head`, `--epochs-finetune`,
`--batch-size`, `--input-size`, `--output`.

## Inference integration

Two changes in `classifier.py`:

1. Module-level lazy load:
   ```python
   _model = None
   def _get_model():
       global _model
       if _model is None:
           _model = tf.keras.models.load_model(MODEL_PATH)
       return _model
   ```
2. Replace stub body:
   ```python
   def predict(squares_batch):
       resized = tf.image.resize(squares_batch, INPUT_SIZE) / 255.0
       probs = _get_model().predict(resized, verbose=0)
       indices = probs.argmax(axis=1)
       return [INDEX_TO_LABEL[i] for i in indices]
   ```
   Add `INDEX_TO_LABEL = ['.', 'P', 'N', ...]` constant.

The stub stays as a fallback when `models/classifier.keras` is missing —
log a one-line warning at import time.

## Evaluation

Two test surfaces:

1. **Per-crop metrics** on held-out validation: top-1 accuracy, per-class
   confusion matrix, weighted F1. Run as the last step of `train.py`
   and dump to `models/eval.json`.
2. **Per-position metrics** on a held-out set of full board images:
   for each board, run `segment_board → predict → labels_to_placement`
   and compare to ground-truth FEN. Report position-accuracy
   (all-64-correct) and per-position label-edit distance.

Position-accuracy is the metric that matters for the real pipeline.
Per-crop accuracy can look great while position-accuracy is terrible
(see the `0.95**64` calculation above).

## Open decisions (revisit before writing code)

1. **Color model.** Is `empty` one class or three (light-square-empty,
   dark-square-empty, just-empty-no-color-cue)? Single class is simpler
   and what's planned above. Three classes would let the model use the
   square color as a sanity check but increases the label set to 15.
2. **Input size 64 vs 96.** MobileNetV2 prefers ≥ 96. If we go 96, our
   `segment_board` still emits 64×64 (assuming a 512px board) — the
   `tf.image.resize` in inference handles upscale, but it'd be cleaner
   to use `--board-size 768` in calibration so 96×96 native crops come
   straight out. Decision: probably 96.
3. **Synthetic pretrain or skip.** Empirical question — try both, keep
   what works. If self-capture is fast enough to get 2000 crops, skip
   synthetic entirely.
4. **Active learning loop.** Once v0 is trained, use it to predict
   crops we haven't labeled yet; manually correct only the
   low-confidence or wrong ones. This is the cheapest way to get from
   v0 to v1.

## Order of operations (when we do start coding)

1. Scaffold `data/` folder structure + a tiny `tools/label_helper.py`
   that lists crops needing classification and lets you drop each into
   a folder with one keypress.
2. Run `main.py --save-squares` on 20-30 captured positions → ~1500
   crops to label.
3. Label them. Live with the boredom.
4. Write `train.py`, run phase 1 + phase 2.
5. Drop the stub from `classifier.py`, point `_get_model` at the saved
   `.keras` file.
6. Run the live pipeline end-to-end on a real game. Watch
   `[move NNN] no legal move matches predicted position` go from
   100% to 5% to 0%.
7. (Optional) Synthetic pretrain pass if real-only doesn't get there.
