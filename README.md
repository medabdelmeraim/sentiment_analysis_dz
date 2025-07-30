# 🧠 Arabic Sentiment Analysis DZ 🇩🇿

This project is a **sentiment analysis application** for Algerian Arabic (Darija), designed to classify text into **Positive**, **Negative**, or **Neutral** categories. It supports both real-time input and batch prediction through a CSV upload using a clean **Streamlit web interface**.

## 🛠️ Models Used

We trained and compared several deep learning models on concatenated Algerian Arabic sentiment datasets:

| Model                        | Accuracy (Test) |
|-----------------------------|------------------|
| ✅ LSTM                     | **~82%**         |
| 🧠 CNN                      | ~80%             |
| 🔀 Transformer + LSTM      | ~71%             |

All models were trained on **multiple combined Algerian sentiment datasets**, including dialectal and social media data.

## ⚙️ Features

- Text preprocessing (cleaning, stemming, stopword removal)
- Support for Arabic dialect (Darija)
- Real-time text classification
- Batch predictions with CSV upload
- Word cloud visualization per sentiment
- Downloadable result files


