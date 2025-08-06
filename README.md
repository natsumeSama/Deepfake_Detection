# 🕵️ Deepfake Detection (PyTorch)

A binary image classification project to detect deepfake images using Convolutional Neural Networks (CNN).
The project supports both **RGB** and **Grayscale** models and includes data downloading, preprocessing, training, evaluation, and model saving.

---

## 📂 Project Structure

```
Deepfake_Detection/
│
├── data_setup.py          # Dataset loading & preprocessing
├── model_builder.py       # CNN model definition
├── engine.py              # Training and evaluation loops
├── utils.py               # Utility functions (e.g. save_model)
├── train.py               # Train and save RGB & Grayscale models
├── evaluate.py            # Evaluate models on test set
├── download_data.py       # Download and prepare dataset
├── requirements.txt       # Project dependencies
└── models/                # Saved models (after training)
```

---

## 🚀 Features

- ✅ Dataset download from Google Drive
- ✅ Custom `Dataset` class with OpenCV and Torch transforms
- ✅ Training on both RGB and Grayscale images
- ✅ Model selection based on best score (loss + accuracy)
- ✅ Evaluation script for testing saved models
- ✅ Modular structure for scalability

---

## 🛠️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/Deepfake_Detection.git
   cd Deepfake_Detection
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 📥 Dataset

The dataset is automatically downloaded via `download_data.py`.

```bash
python download_data.py
```

This script:

- Checks if the dataset is already present
- Downloads it from Google Drive
- Extracts it into `data/deepfakes/db/`

---

## 🏋️ Training

To train both RGB and Grayscale models:

```bash
python train.py
```

This will:

- Train CNN models on both RGB and Grayscale data
- Save them in the `models/` directory

---

## 📊 Evaluation

To evaluate the trained models:

```bash
python evaluate.py
```

Example output:

```
📊 Starting evaluation for all models...

✅ CNN_RGB.pth → Loss: 0.2451 | Accuracy: 0.9234
✅ CNN_GRAY.pth → Loss: 0.2873 | Accuracy: 0.9012

📌 Summary:
 - RGB: Loss=0.2451 | Accuracy=0.9234
 - Grayscale: Loss=0.2873 | Accuracy=0.9012
```

---

## 🏗️ Tech Stack

- [Python 3.10+](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [OpenCV](https://opencv.org/)
- [gdown](https://pypi.org/project/gdown/)

---

## 📌 TODO (Future Improvements)

- [ ] Add data augmentation pipeline
- [ ] Implement early stopping during training
- [ ] Integrate confusion matrix and classification report
- [ ] Experiment with transfer learning models (e.g. ResNet)

---
