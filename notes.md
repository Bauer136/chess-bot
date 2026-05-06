Pipeline:

**Stage 1 — Video capture** (OpenCV / cv2.VideoCapture)
OpenCV is the workhorse for reading camera or video file input. cv2.VideoCapture opens the stream (USB cam, IP camera, or file), and you control FPS, resolution, and frame buffering here. This is the entry point of the entire pipeline.

**Stage 2 — Frame preprocessing** (OpenCV + NumPy)
Raw frames are noisy and inconsistently sized. OpenCV handles resizing, color space conversion (BGR→RGB), and Gaussian blur to reduce noise. NumPy normalizes pixel values to [0,1] floats that the CNN expects. This stage also handles perspective correction if the camera isn't perfectly overhead.

**Stage 3 is now a temporal move detector**, not just a board localizer. In the original pipeline, every frame triggered the full analysis chain — CNN inference, FEN reconstruction, engine call, TTS output — which is wasteful and noisy. With C3D processing a sliding 16-frame clip, the model learns the actual spatiotemporal signature of a chess move: a hand entering the frame, lifting a piece, translating, placing, withdrawing. Nothing downstream fires until C3D is confident a move genuinely occurred. This eliminates false triggers from shadows, wobbling pieces, and clock-hitting, which are the main sources of noise in live chess capture.

**Stage 4 — Square segmentation** (NumPy + tf.data)
Once the board is localized and rectified, NumPy slices it into a 8×8 grid of 64 square crops. These are batched into a tf.data.Dataset pipeline for efficient, prefetched feeding into the CNN — critical for real-time performance.

**Stage 5 — CNN piece classifier** (TensorFlow / Keras)
The core model. A CNN built with tf.keras using Conv2D, MaxPooling2D, BatchNormalization, and Dropout layers classifies each square into one of 13 classes (6 white pieces, 6 black pieces, empty). Input is ~64×64px per square. Output is a softmax probability vector.

**Stage 6 — Training pipeline** (TensorFlow Datasets + transfer learning)
You won't collect enough data from scratch. Use a pre-trained backbone like MobileNetV2 or EfficientNetB0 (available in tf.keras.applications) and fine-tune on a chess piece dataset. ImageDataGenerator handles augmentation (rotations, brightness shifts, flips) to improve robustness across different board sets and lighting conditions.

**Stage 7 — Board state reconstruction** (python-chess)
The 64 CNN predictions are assembled into a FEN string — the standard notation for a chess position. python-chess validates move legality, tracks whose turn it is, and handles castling/en passant state. This is the bridge between computer vision and chess logic.

**Stage 8 — Chess engine** (Stockfish + python-chess UCI bridge)
python-chess provides a clean engine.play() interface to Stockfish via the UCI protocol. You set search depth or time limits, and Stockfish returns the best move, evaluation score, and optionally multiple lines. This is the "brain" of the system.

**Stage 9 — Move-to-speech formatting** (python-chess + string templates)
Stockfish returns a move like e2e4 or g1f3. python-chess converts this to Standard Algebraic Notation (e4, Nf3). A small template layer then turns it into natural speech — "Knight to F3" or "Pawn takes E4" — ready for the TTS engine.

**Stage 10 — Text-to-speech** (pyttsx3 or gTTS + pygame)
pyttsx3 runs entirely offline and speaks through the system audio device with no latency from API calls — best for real-time. gTTS produces higher-quality audio but requires network access and adds ~500ms of latency. pyaudio gives you direct control over which output device (speaker) is used.

**Stage 11 — Orchestration & runtime** (asyncio + threading + queue)
The camera loop, CNN inference, engine calls, and TTS output all run at different speeds. Python's queue.Queue decouples them so a slow Stockfish search doesn't drop frames. asyncio or a thread-per-stage architecture keeps everything flowing. argparse handles CLI config; Docker containerizes the whole stack for deployment.

**Stage 12 — Evaluation & monitoring** (TensorBoard + sklearn + W&B)
TensorBoard tracks training loss and accuracy curves. sklearn.metrics gives you a per-class confusion matrix (essential — you need to know if it's confusing bishops and queens). An imshow debug overlay in OpenCV draws bounding boxes and predictions on-screen during development. Weights & Biases is optional but excellent for experiment tracking if you iterate on the model architecture.