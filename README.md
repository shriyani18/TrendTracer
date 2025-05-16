# 🚀 Trend Tracer

![GitHub repo size](https://img.shields.io/github/repo-size/shriyani18/TrendTracer?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-green?style=for-the-badge)

---

**GitHub Repository:** [Trend Tracer](https://github.com/shriyani18/TrendTracer)

---

### 🔍 What is Trend Tracer?

Trend Tracer is a **deep learning powered image similarity search engine** that helps you find trending and visually similar images from a large dataset!  
Built on top of **ResNet50** with a sleek **Streamlit** interface, this tool is perfect for visual trend analysis and recommendation.

---

### ⚙️ Features

- Extracts **2048-dimensional image features** using pretrained ResNet50 🖼️
- Uses **Global Max Pooling** to create robust feature vectors
- Employs **Nearest Neighbors (Euclidean distance)** to find similar images 🔎
- Supports **44,000+ images** dataset for comprehensive trend tracking
- User-friendly **Streamlit app** for image upload and instant recommendations
- Saves and loads embeddings for super-fast search 🚀

---

### 🛠️ Installation & Setup

```bash
git clone https://github.com/yourusername/trend-tracer.git
cd trend-tracer
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
