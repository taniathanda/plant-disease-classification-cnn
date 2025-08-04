# 🌿 Plant Leaf Disease Detection Using Deep Learning

This project implements a **Convolutional Neural Network (CNN)** & **ResNet-18(Transfer Learning)** using PyTorch to classify plant leaf images into different disease categories. The model is trained on the **PlantVillage dataset** and achieves high accuracy through proper data augmentation, normalization, and model tuning.

---

## 📁 Dataset
- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- 15 categories with thousands of high-quality images
- Preprocessed with resizing, normalization, and augmentation

---

## 🛠️ Methods Used
- PyTorch and torchvision
- Image transformations and normalization
- ResNet-18 (Transfer Learning)
- Custom CNN with:
  - 3 convolutional blocks
  - BatchNorm, ReLU, Dropout
  - Fully connected layers
- Metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrix visualization

---

## 📊 Results
- Custom CNN
-   •	Accuracy: 96%
    •	Weighted Average Precision: 0.96
    •	Weighted Average Recall: 0.96
    •	Weighted Average F1-score: 0.96
-ResNet-18 (Transfer Learning)
  •	Accuracy: 98%
  •	Weighted Average Precision: 0.98
  •	Weighted Average Recall: 0.98
  •	Weighted Average F1-score: 0.98  

---

## 📌 How to Run
1. Download the jupyter notebook and run in Google Colab or Jupyter Notebook
2. Install required libraries:
   ```bash
   pip install torch torchvision matplotlib seaborn scikit-learn
