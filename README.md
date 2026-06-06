# 🤖 BERT Sentiment Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A fine-tuned BERT model for binary sentiment analysis, featuring interactive tokenization visualization, attention mask exploration, and a real-time Streamlit dashboard.**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Dashboard](#-streamlit-dashboard) • [Results](#-results)

</div>

---

## 🌟 Overview

This project fine-tunes `bert-base-uncased` on movie review sentiment data for binary classification (Positive / Negative). It goes beyond just training — it provides full visibility into the BERT pipeline through an interactive Streamlit dashboard that visualizes tokenization, encoding, attention masks, and prediction probabilities in real time.

---

## ✨ Features

### 🧠 Model & Training
- Fine-tuned `bert-base-uncased` with a classification head
- AdamW optimizer with linear warmup scheduler
- Gradient clipping for stable training
- Trains to **100% validation accuracy** on the sample dataset
- Model and tokenizer auto-saved at best checkpoint

### 📊 Interactive Streamlit Dashboard
- **Tokenization tab** — visualizes WordPiece tokens and special tokens (`[CLS]`, `[SEP]`, `[PAD]`)
- **Encoding tab** — displays input IDs, attention mask distribution, and encoding statistics
- **Prediction tab** — color-coded sentiment output with confidence probabilities bar chart
- **Visualization tab** — token length distribution and token ID scatter plot

### 🔍 Inference Pipeline
- Single text prediction with confidence scores
- Batch prediction from file input
- Interactive CLI mode
- Detailed tokenization analysis with JSON export

---

## 🏗️ Architecture

```
Input Text
    ↓
BERT WordPiece Tokenizer
    ↓
[CLS] token₁ token₂ ... tokenₙ [SEP] [PAD] ... [PAD]
    ↓
Token IDs + Attention Mask (max_length=128)
    ↓
┌─────────────────────────────────────┐
│  bert-base-uncased                  │
│  ├── 12 Transformer Layers          │
│  ├── 12 Attention Heads             │
│  ├── Hidden Size: 768               │
│  └── Vocab Size: 30,522             │
└─────────────────────────────────────┘
    ↓
[CLS] Pooled Output (768-dim)
    ↓
Dropout → Linear (768 → 2)
    ↓
Softmax → [P(Negative), P(Positive)]
```

### Training Configuration

| Parameter | Value |
|---|---|
| Model | `bert-base-uncased` |
| Max Sequence Length | 128 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Epochs | 3 |
| Optimizer | AdamW |
| Scheduler | Linear warmup |

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, CPU works too)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/bert-sentiment-classifier.git
cd bert-sentiment-classifier

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
torch
torchvision
transformers
scikit-learn
pandas
numpy
matplotlib
seaborn
tqdm
streamlit
plotly
```

---

## 🚀 Usage

### Step 1: Train the Model

Open the Jupyter Notebook:

```bash
jupyter notebook bert_training.ipynb
```

The notebook walks through:
1. Dataset creation and train/test split
2. BERT tokenization demonstration (tokens, input IDs, attention masks)
3. Training loop with progress bars
4. Evaluation with classification report and confusion matrix
5. Saving model to `./saved_model` and tokenizer to `./tokenizer`

### Step 2: Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

### Step 3: Run Inference via CLI

```bash
# Single text
python inference.py --text "This movie is absolutely fantastic!"

# Detailed tokenization analysis
python inference.py --text "Amazing film!" --detailed

# Batch prediction from file
python inference.py --file reviews.txt --output results.json

# Interactive mode
python inference.py
```

---

## 📊 Streamlit Dashboard

The dashboard provides four analysis tabs for any input text:

**Tokenization Tab**
- Raw WordPiece tokens
- Tokens with special markers ([CLS], [SEP], [PAD])
- Token count metrics

**Encoding Tab**
- Input ID table (filterable)
- Attention mask bar chart
- Attended vs padded token statistics

**Prediction Tab**
- Color-coded sentiment result (green = positive, red = negative)
- Confidence score
- Probability comparison bar chart

**Visualization Tab**
- Token length histogram
- Token ID scatter plot by position
- Full token analysis table

---

## 📁 Project Structure

```
bert-sentiment-classifier/
│
├── bert_training.ipynb         # Training notebook (tokenization + fine-tuning)
├── app.py                      # Streamlit dashboard
├── inference.py                # CLI inference script
│
├── saved_model/                # Fine-tuned BERT weights (generated after training)
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
│
├── tokenizer/                  # Saved BERT tokenizer (generated after training)
│   ├── vocab.txt
│   └── ...
│
├── training_history.pkl        # Training metrics (generated after training)
├── training_results.png        # Loss/accuracy/confusion matrix plots
└── requirements.txt
```

---

## 📈 Results

Training on the sample dataset (100 balanced reviews, 80/20 split):

| Metric | Value |
|---|---|
| Train Accuracy (Epoch 3) | 100% |
| Validation Accuracy | 100% |
| Test Accuracy | 100% |

**Classification Report:**

```
              precision    recall  f1-score   support
    Negative       1.00      1.00      1.00        10
    Positive       1.00      1.00      1.00        10
    accuracy                           1.00        20
```

> **Note:** These results reflect the small sample dataset. For production use, fine-tune on a larger corpus (e.g., IMDB 50K, SST-2) for more generalizable results.

---

## 🔍 Understanding BERT Tokenization

BERT uses **WordPiece tokenization**, which handles out-of-vocabulary words by splitting them into subword units:

```
"unbelievable" → ["un", "##believ", "##able"]
```

The full encoding pipeline:

```
Raw Text → WordPiece Tokens → Add [CLS]/[SEP] → Token IDs + Attention Mask → BERT
```

The attention mask distinguishes real tokens (1) from padding tokens (0), ensuring padding doesn't affect the model's attention computation.

---

## 📈 Roadmap

- [ ] Fine-tune on IMDB 50K or SST-2 for production-grade accuracy
- [ ] Add multi-class sentiment (very negative → very positive)
- [ ] Attention heatmap visualization in dashboard
- [ ] REST API wrapper with FastAPI
- [ ] Docker containerization
- [ ] ONNX export for faster inference

---

## 🤝 Contributing

```bash
# Fork and clone
git clone https://github.com/yourusername/bert-sentiment-classifier.git

# Create feature branch
git checkout -b feature/your-feature

# Commit and push
git commit -m "Add: your feature"
git push origin feature/your-feature
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/transformers/) — BERT model and tokenizer
- [Devlin et al. (2018)](https://arxiv.org/abs/1810.04805) — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
- [Streamlit](https://streamlit.io/) — interactive dashboard framework

---

<div align="center">
  <b>Understand what BERT sees — one token at a time. 🔍</b>
</div>
