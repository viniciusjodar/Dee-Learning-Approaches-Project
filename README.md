<<<<<<< HEAD
# Dee-Learning-Approaches-Project
=======
# 🧠 Deep Learning Approaches

This repository contains two main notebooks exploring different Deep Learning techniques:

1. **`Cnn_project.ipynb`** – Implementation of a Convolutional Neural Network (CNN) from scratch using TensorFlow/Keras to classify images from the **Fashion-MNIST** dataset.  
2. **`transfer_learning.ipynb`** – Application of **Transfer Learning** with pre-trained models (e.g., VGG16, ResNet50) for image classification, leveraging networks already trained on ImageNet.

---

## ⚡ Requirements

- Windows 10/11 with **WSL2 (Ubuntu)** configured  
- NVIDIA GPU (**RTX 3050** in this case)  
- NVIDIA drivers installed on Windows (`nvidia-smi` should work)  
- VS Code with the following extensions:
  - **Remote – WSL**
  - **Python**
  - **Jupyter**

---

## 🛠️ GPU Environment Setup

### 1. Create a virtual environment in WSL
```bash
# inside Ubuntu/WSL
sudo apt update && sudo apt upgrade -y

# create the environment
python3 -m venv ~/tf220

# activate the environment
source ~/tf220/bin/activate
```

### 2. Install TensorFlow with GPU support
```bash
pip install --upgrade pip
pip install "tensorflow[and-cuda]==2.20.0"
```

✅ This version includes CUDA and cuDNN, no manual installation required.

### 3. Register the environment in Jupyter
```bash
pip install ipykernel jupyter
python -m ipykernel install --user --name tf220 --display-name "Python (tf220)"
```

Now the kernel `Python (tf220)` will be available in VS Code and Jupyter.

### 4. Verify GPU availability
```python
import tensorflow as tf
print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))
```

Expected output:
```
TF: 2.20.0
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## ▶️ How to Run the Notebooks

1. Open **Ubuntu/WSL** and navigate to the project folder:
   ```bash
   cd ~/projects/deep-learning-approaches
   code .
   ```
   (This launches VS Code connected to WSL2.)

2. In the notebook (`.ipynb`), select the kernel **Python (tf220)**.

3. Run all cells (`Run All`).

---

## 📂 Project Structure

### 1. `Cnn_project.ipynb`
- Preprocessing the Fashion-MNIST dataset  
- Building a CNN:
  - Conv2D + MaxPooling layers  
  - Flatten + Dense layers  
- Training with the **Adam optimizer**  
- Callbacks:
  - `ModelCheckpoint`  
  - `EarlyStopping`  
- Evaluation with **accuracy** and **loss** metrics  
- Plotting training history curves

### 2. `transfer_learning.ipynb`
- Load a pre-trained model (e.g., VGG16)  
- Freeze the base layers  
- Add dense layers for custom classification  
- Optional fine-tuning of top layers  
- Compare performance between **CNN from scratch** and **Transfer Learning**

---

## 📊 Expected Results
- CNN from scratch (Fashion MNIST): accuracy around **88–92%**  
- Transfer Learning: higher accuracy on complex datasets with less training  

---

## 📌 Tips
- Monitor GPU usage in real-time:
  ```bash
  watch -n 1 nvidia-smi
  ```
- To optimize performance, experiment with:
  - `batch_size`
  - `learning_rate`
  - `patience` in `EarlyStopping`

---

✍️ Authors: *Your Name, Kaan, Jithin*  
📅 Date: 2025
>>>>>>> dbc1f73 (Initial commit: Deep Learning Approaches (CNN + Transfer Learning + README))
