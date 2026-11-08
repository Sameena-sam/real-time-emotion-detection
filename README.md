# Real-Time Emotion Detection

A convolutional neural network that classifies seven facial emotions (angry, disgust, fear, happy, neutral, sad, surprise) from 48x48 greyscale images, with a script that runs it on a live webcam feed.

## Results

Measured on the 7,178-image test set with `models/final_emotion_detection_model.keras`:

| Metric | Value |
|---|---|
| Accuracy | 56.4% (always guessing the most common emotion scores 24.7%) |
| Recall: happy / surprise / neutral | 0.76 / 0.70 / 0.61 |
| Recall: sad / angry / fear / disgust | 0.48 / 0.45 / 0.31 / 0.23 |
| Processing time per frame on CPU | about 55 ms (preprocessing and model, not counting camera capture and display) |

Happy and surprise are recognised best; disgust, the rarest class, is hardest. The test set was also used for early stopping, so accuracy may be slightly optimistic.

## Model

Three convolution and max-pooling blocks (32, 64 and 128 filters), a 128-unit dense layer with 50% dropout, and a 7-way softmax. Trained with Adam on 28,709 images, with early stopping and checkpointing on validation loss.

## Project structure

```
real-time-emotion-detection/
├── src/
│   ├── train_model.py                  # builds, trains and evaluates the CNN
│   └── real_time_emotion_detection.py  # webcam demo
├── models/
│   └── final_emotion_detection_model.keras
├── data/                               # training arrays (not included, about 660 MB)
├── requirements.txt
└── README.md
```

## Run

From the repository root:

```bash
pip install -r requirements.txt
python src/real_time_emotion_detection.py   # press q to quit
```

To retrain, put `x_train.npy`, `y_train.npy`, `x_test.npy` and `y_test.npy` (48x48 greyscale images scaled to 0 to 1, integer labels 0 to 6) in `data/`, then run `python src/train_model.py`.

## Next steps

The webcam script classifies the whole frame. Detecting the face first and classifying only that region would likely improve live results.

## Tech

Python, TensorFlow/Keras, OpenCV, NumPy.

## Author

Sameena Siddiqua
