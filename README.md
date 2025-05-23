# 🧠 DCGAN – Handwritten Digit Generator

This project implements a Deep Convolutional Generative Adversarial Network (**DCGAN**) using **PyTorch** to generate realistic handwritten digits based on the MNIST dataset. The project includes a fully functional **Streamlit app** that allows users to generate digits or simulate retraining the generator using their own uploaded images.

---

## 🚀 Live Demo

🔗 *\[Add your Streamlit Cloud or Hugging Face Spaces link here]*

---

## 🧾 Project Overview

* **Model Type:** Deep Convolutional GAN
* **Dataset:** MNIST (handwritten digits)
* **Libraries:** PyTorch, TorchVision, Streamlit
* **Output:** 28x28 grayscale digit images generated from random noise

---

## 📁 Folder Structure

```
DCGAN-Digit-Generator/
├── generator_model.py           # Generator class definition
├── generator.pth                # Trained generator weights
├── dcgan_streamlit_app.py       # Streamlit app
├── requirements.txt             # App dependencies
└── README.md                    # Project documentation
```

---

## ⚙️ Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/DCGAN-Digit-Generator.git
cd DCGAN-Digit-Generator

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run dcgan_streamlit_app.py
```

---

## 🧠 Features

* 🎲 Generate batches of digits using random noise
* 📤 Upload a grayscale image to simulate retraining the generator
* 📸 Real-time visualization of generated samples

---

## 📊 Tools & Libraries

| Category      | Tool                          |
| ------------- | ----------------------------- |
| Deep Learning | PyTorch                       |
| Data Handling | torchvision.datasets (MNIST)  |
| Visualization | matplotlib, torchvision.utils |
| Web Interface | Streamlit                     |
| Image Upload  | PIL, torchvision.transforms   |

---

## 🧠 Project Insights & Conclusion

* **DCGANs** can generate realistic digit images after a few epochs of training.
* **Generator stability** is critical, and balancing losses is key to success.
* The app makes **GAN concepts approachable** through visual and interactive experimentation.

### 🔍 Key Takeaway

> Generative models like DCGANs are powerful, and building interactive demos helps communicate their potential clearly and effectively.

---

## 👨‍💻 Author


**Youssef Bassiony**
[LinkedIn](https://www.linkedin.com/in/youssef-bassiony) | [GitHub](https://github.com/youssef-113)

---

⭐ If you like this project, give it a star and share your feedback!
