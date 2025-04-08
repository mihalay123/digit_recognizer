# 🧠 Digit Recognizer – Real-time Handwritten Digit Detection

This is a Python project that uses a trained neural network (Keras + TensorFlow) to recognize handwritten digits in real-time using your webcam.

## ✨ Features

- 📷 Real-time digit detection from webcam
- 🎯 Digit detection inside a movable frame
- ✅ Uses trained MNIST model (28x28 grayscale)
- 🧼 Clean terminal output (no OpenCV camera noise)
- 🛠 Works on macOS, Windows, and Linux (platform-aware)

---

## 📦 Requirements

- Python 3.8+
- [TensorFlow](https://www.tensorflow.org/install)
- OpenCV (`opencv-python`)
- NumPy

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Clone the repository (or copy files locally)
2. Make sure your trained model is saved as `models/digit_model.h5`
3. Run the main script:

```bash
python main.py
```

4. Choose a camera index (you will see available options printed in terminal)
5. Show a handwritten digit inside the blue square on screen

Press **`Q`** to quit.

---

## 🧠 How It Works

- The webcam feed is captured in real time using OpenCV
- A central square region (300x300px) is extracted as the **Region of Interest (ROI)**
- Image is preprocessed: grayscale → blur → adaptive threshold → resize → normalize
- The resulting 28×28 image is passed into a trained MNIST model
- The predicted digit is shown above the square

---

## 📁 Project Structure

```
digit_recognizer/
│
├── models/
│   └── digit_model.h5         # trained Keras model
│
├── main.py                    # real-time digit recognizer
├── requirements.txt
└── README.md
```

---

## ⚙️ Training Your Own Model (optional)

If you'd like to train your own model:

```bash
cd scripts
python train_model.py
```

The model will be saved in `models/digit_model.h5`.

---

## 💡 Tips

- Use a bold black marker on white background for best results
- Good lighting improves detection
- Works better with digits similar in style to MNIST

---

## 🧪 Tested On

- ✅ macOS Ventura + iPhone (Continuity Camera)
- ✅ macOS with built-in webcam
- ✅ Windows 10 + USB webcam
- ✅ Ubuntu 22.04 + Logitech webcam

---

## 📄 License

This project is MIT licensed. Feel free to use and modify it for your own experiments.
