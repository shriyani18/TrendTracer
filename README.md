# 🚀 Trend Tracer

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
git clone https://github.com/shriyani18/TrendTracer.git
cd TrendTracer
python -m venv venv
source venv/bin/activate  # (Use `venv\Scripts\activate` on Windows)
pip install -r requirements.txt

