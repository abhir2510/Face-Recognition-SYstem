# Facial Recognition System Using Deep Learning

A small, local facial recognition project built around the pipeline requested in the mini-project report: **OpenCV -> MTCNN -> FaceNet -> classifier**.

> Responsible-use note: enroll only people who have explicitly agreed to participate. This project is intended for a local classroom demonstration, not covert surveillance or an identity decision with real-world consequences.

## What it does

1. **Enroll a person**: capture several webcam frames, detect the face, and save the averaged FaceNet embedding under a name.
2. **Recognize faces**: read the webcam, detect every face, create an embedding, and compare it with enrolled embeddings.
3. **Show a result**: draw a bounding box, name, and confidence-like similarity score. If there is no sufficiently close match, display `Unknown`.

## Tech stack

| Layer | Technology | Role |
|---|---|---|
| Language | Python 3.10+ | Main application logic |
| Video and UI | OpenCV | Webcam frames, colour conversion, drawing windows |
| Face detection | MTCNN | Finds face bounding boxes and five landmark points |
| Deep learning | TensorFlow + Keras-FaceNet | Generates a normalized 128-dimensional FaceNet embedding |
| Recognition | NumPy cosine similarity | Matches the embedding to enrolled identities |
| Optional evaluation | scikit-learn | Precision, recall, F1-score, confusion matrix |

This uses **pre-trained** detection and embedding models; it does not train a biometric model from zero. That keeps the mini-project practical on an ordinary laptop.

## Setup

Create a virtual environment and install packages:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Run the commands from this project folder.

## Demo workflow

Enroll each consenting participant (collect 20-40 samples with slight head/lighting changes):

```bash
python enroll.py --name Abhishek --samples 30
python enroll.py --name Aditya --samples 30
```

Then launch recognition:

```bash
python recognize.py
```

Controls: `q` quits; `c` captures a usable enrollment sample. The model weights may download on their first run, so keep an internet connection available for that initial setup.

## How it works - explanation for a viva

"The webcam gives us frames through OpenCV. MTCNN first locates each face and its landmark points, so background pixels are not treated as a face. The detected face is resized to 160 by 160 pixels and sent to FaceNet. FaceNet converts the image into 128 numbers called an embedding. Faces of the same person have embeddings close together; different people are farther apart. During enrollment, we average multiple embeddings for each name. During recognition, we calculate cosine similarity between the live embedding and every saved average. If the best similarity passes a threshold, we display that name; otherwise we say Unknown."

### Pipeline

```text
Webcam frame
   -> MTCNN face detection + landmarks
   -> crop / resize to 160 x 160
   -> FaceNet deep CNN
   -> 128-D normalized embedding
   -> cosine similarity against enrolled embeddings
   -> known identity or Unknown
```

### Why deep learning is used

Traditional methods hand-design features such as edges or distances between facial parts. FaceNet learns a much richer representation from many training faces. The resulting embedding is compact, so matching is fast even on a CPU.

## Adjustable threshold

`recognize.py` uses a default cosine-similarity threshold of `0.72`. Increase it to reduce false matches (but get more `Unknown` results); lower it to accept more matches (but risk wrong names). Tune it with consented test images rather than claiming one threshold works for everyone.

## Testing ideas for the report

Test the system with held-out, consented images under normal light, low light, a side pose, glasses/mask, and no face. Record true positives, false positives, false negatives, and latency/FPS. Then calculate precision, recall, and F1 using `evaluate.py` or scikit-learn.

## Limitations and future scope

- A face similarity score is not proof of identity.
- Performance can drop with poor lighting, occlusion, extreme pose, and demographic or dataset bias.
- This baseline has no liveness detection, so do not use it for authentication. Future work: anti-spoofing, calibrated thresholds, encrypted embedding storage, access control, and edge-model optimization.
