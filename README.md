# 🧠 MindPulse AI — Emotion Intelligence Platform

MindPulse AI is an advanced AI-powered Emotion and Sentiment Analysis platform built using Hugging Face Transformers, PyTorch, and Streamlit. The system analyzes user text, detects emotions, performs sentiment analysis, understands emojis and slang, and visualizes emotional trends through an interactive dashboard.

The project combines Natural Language Processing (NLP), deep learning, and real-time analytics to create a smart emotion intelligence system capable of understanding human emotional patterns from text inputs.

---

# 🚀 Features

## ✅ Emotion Detection
Detects emotions from user text using Transformer-based deep learning models.

## ✅ DistilBERT Emotion Classification
Uses a fine-tuned DistilBERT Transformer model for accurate emotion prediction from text inputs.

## ✅ Sentiment Analysis
Classifies text sentiment into:
- Positive
- Negative
- Neutral

## ✅ Emoji-Aware NLP
Understands emotional meaning behind emojis and integrates them into predictions.

## ✅ Slang Normalization
Converts internet slang and informal text into meaningful language for improved NLP performance.

## ✅ Vocabulary Learning System
Tracks unknown words and dynamically expands vocabulary understanding.

## ✅ Real-Time Confidence Scores
Displays confidence probability for predictions.

## ✅ Analytics Dashboard
Visualizes:
- Emotion trends
- Sentiment distribution
- Prediction history
- User analytics

## ✅ SQLite Database Integration
Stores prediction history and analytics locally.

---

# 🛠️ Tech Stack

## Frontend
- Streamlit

## Backend
- Python

## AI / Machine Learning
- DistilBERT (Transformer-based NLP Model)
- Hugging Face Transformers
- PyTorch
- Scikit-learn

## NLP & Sentiment Analysis
- VaderSentiment
- Emoji Processing
- Tokenization & Text Normalization

## Database
- SQLite

## Data Visualization
- Plotly

---

# 📂 Project Structure

```plaintext
MindPulse-AI/
│
├── mindpulse_model/
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── label_encoder.pkl
│   └── unknown_words.json
│
├── src/
│   ├── database.py
│   ├── emoji_utils.py
│   ├── emoji_wrapper.py
│   ├── predict.py
│   ├── sentiment.py
│   ├── slang.py
│   └── vocab_updater.py
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── run.bat
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/tanmay-005/mindpulse-ai.git
```

---

## 2️⃣ Move Into Project Folder

```bash
cd mindpulse-ai
```

---

## 3️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
```

---

## 4️⃣ Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

The application will launch in your browser.

---

# 🧠 AI Architecture

The platform uses a DistilBERT-based Transformer NLP model fine-tuned for emotion classification and sentiment understanding.
### Pipeline:
1. Text Input
2. Slang Normalization
3. Emoji Interpretation
4. Tokenization
5. Transformer Inference
6. Emotion Prediction
7. Sentiment Classification
8. Analytics Storage

---

# 📊 Dashboard Features

- Emotion Frequency Charts
- Sentiment Distribution
- Confidence Score Visualization
- Historical Predictions
- Real-Time Analytics

---

# 🔮 Future Enhancements

- 🎤 Voice Emotion Detection
- 🌍 Multi-language Support
- 😀 Facial Emotion Recognition
- 🤖 AI Chat Assistant
- 📱 Mobile Application
- ☁️ Cloud Deployment
- 📈 Advanced Deep Learning Models

---

# 📌 Important Note

Large model weight files (`.safetensors`) are excluded from GitHub due to GitHub file-size limitations.

To use custom trained models:
- Place model files inside:
  
```plaintext
mindpulse_model/
```

---

# 👨‍💻 Author

**Tanmay Rana**

---

# 📜 License

This project is intended for educational, research, and portfolio purposes.

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
