# 🌿 Plant Disease Classification using CNN

This project implements a **Convolutional Neural Network (CNN)** using PyTorch to classify plant leaf images into different disease categories. The model is trained on the **PlantVillage dataset** and achieves high accuracy through proper data augmentation, normalization, and model tuning.

---

## 📁 Dataset
- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- 15 categories with thousands of high-quality images
- Preprocessed with resizing, normalization, and augmentation

---

## 🛠️ Methods Used
- PyTorch and torchvision
- Image transformations and normalization
- Custom CNN with:
  - 3 convolutional blocks
  - BatchNorm, ReLU, Dropout
  - Fully connected layers
- Metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrix visualization

---

## 📊 Results
- **Validation Accuracy**: ~95%
- **Test Accuracy**: ~96%
- Visual results and graphs included

---

## 📌 How to Run
1. Install required libraries:
   ```bash
   pip install torch torchvision matplotlib seaborn scikit-learn
