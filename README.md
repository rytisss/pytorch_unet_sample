# PyTorch Autoencoder Sample with NUCLIO

This repository provides a minimal example of using a Segformer model in PyTorch for image segmentation.

## 🧠 Overview

- Runs a simple UNet-based prediction script using a sample image.
- No arguments required — just run the script.
- Includes example image and requirements for setup.

## 📁 Files

- `predict.py`: Runs prediction using a PyTorch UNet model.
- `sample.jpg`: Example input image used by the script.
- `requirements.txt`: Dependencies.

## ⚙️ Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/rytisss/pytorch_unet_sample.git
   cd pytorch_unet_sample
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Run

Simply run the script:

```bash
python predict.py
```

The script uses `sample.jpg` as input and outputs the result or logs it.

## 📄 License

MIT License
