import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from collections import Counter
import matplotlib.pyplot as plt

# ------------------- Setup -------------------
stemmer = ISRIStemmer()
arabic_stopwords = set(stopwords.words("arabic"))

def clean_and_tokenize(text):
    text = re.sub(r'[^\u0621-\u064A\s]', '', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return [stemmer.stem(w) for w in text.split() if w not in arabic_stopwords]

def encode(tokens, vocab, max_len=100):
    ids = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long)

# ------------------- Load Vocabulary -------------------
with open("tokenizer_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# ------------------- Load Label Encoder -------------------
with open("label encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
idx2label = {i: label for i, label in enumerate(label_encoder.classes_)}

# ------------------- CNN Model -------------------
class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 128, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        self.batch_norm = nn.BatchNorm1d(128 * 3)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embedding(x)               # [B, T, E]
        x = x.permute(0, 2, 1)              # [B, E, T]
        x1 = torch.relu(self.conv1(x))      # [B, 128, T]
        x2 = torch.relu(self.conv2(x))
        x3 = torch.relu(self.conv3(x))
        x1 = torch.max(x1, dim=2)[0]        # Global Max Pooling
        x2 = torch.max(x2, dim=2)[0]
        x3 = torch.max(x3, dim=2)[0]
        x = torch.cat([x1, x2, x3], dim=1)  # [B, 128*3]
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# ------------------- Load Model -------------------
device = torch.device("cpu")
model = CNNTextClassifier(vocab_size=len(vocab), embed_dim=100, num_classes=len(idx2label))
model.load_state_dict(torch.load("cnn clasifier.pth", map_location=device))
model.eval()

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Arabic Sentiment CNN", layout="centered")
st.title("🇩🇿 Arabic Sentiment Analysis with CNN")
st.write("Predict the **sentiment** of Arabic text using your custom-trained CNN model.")

# ----------- Text input -----------
st.header("📝 Analyze a Single Sentence")
text_input = st.text_area("Enter Arabic text:")

if st.button("Analyze Text"):
    if text_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        tokens = clean_and_tokenize(text_input)
        encoded = encode(tokens, vocab).unsqueeze(0)
        with torch.no_grad():
            output = model(encoded)
            pred = torch.argmax(output, dim=1).item()
        st.success(f"🧠 Sentiment: **{idx2label[pred]}**")

# ----------- CSV input -----------
st.header("📂 Analyze a CSV File")
uploaded_file = st.file_uploader("Upload a CSV with `ID` and `Commentaire` columns", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if "ID" not in df.columns or "Commentaire" not in df.columns:
            st.error("CSV must contain columns: `ID` and `Commentaire`.")
        else:
            df['tokens'] = df['Commentaire'].astype(str).apply(clean_and_tokenize)
            df['encoded'] = df['tokens'].apply(lambda x: encode(x, vocab))
            inputs = torch.stack(df['encoded'].tolist())

            with torch.no_grad():
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1).numpy()
            df['Statut'] = [idx2label[i] for i in preds]

            # Pie Chart
            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = df['Statut'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')
            st.pyplot(fig)

            st.subheader("📄 Preview of Predictions")
            st.dataframe(df[['ID', 'Commentaire', 'Statut']])

            # Download result
            csv_output = df[['ID', 'Commentaire', 'Statut']].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", data=csv_output, file_name="sentiment_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from collections import Counter
import matplotlib.pyplot as plt

# ------------------- Setup -------------------
stemmer = ISRIStemmer()
arabic_stopwords = set(stopwords.words("arabic"))

def clean_and_tokenize(text):
    text = re.sub(r'[^\u0621-\u064A\s]', '', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return [stemmer.stem(w) for w in text.split() if w not in arabic_stopwords]

def encode(tokens, vocab, max_len=100):
    ids = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long)

# ------------------- Load Vocabulary -------------------
with open("tokenizer_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# ------------------- Load Label Encoder -------------------
with open("label encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
idx2label = {i: label for i, label in enumerate(label_encoder.classes_)}

# ------------------- CNN Model -------------------
class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 128, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        self.batch_norm = nn.BatchNorm1d(128 * 3)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embedding(x)               # [B, T, E]
        x = x.permute(0, 2, 1)              # [B, E, T]
        x1 = torch.relu(self.conv1(x))      # [B, 128, T]
        x2 = torch.relu(self.conv2(x))
        x3 = torch.relu(self.conv3(x))
        x1 = torch.max(x1, dim=2)[0]        # Global Max Pooling
        x2 = torch.max(x2, dim=2)[0]
        x3 = torch.max(x3, dim=2)[0]
        x = torch.cat([x1, x2, x3], dim=1)  # [B, 128*3]
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# ------------------- Load Model -------------------
device = torch.device("cpu")
model = CNNTextClassifier(vocab_size=len(vocab), embed_dim=100, num_classes=len(idx2label))
model.load_state_dict(torch.load("cnn clasifier.pth", map_location=device))
model.eval()

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Arabic Sentiment CNN", layout="centered")
st.title("🇩🇿 Arabic Sentiment Analysis with CNN")
st.write("Predict the **sentiment** of Arabic text using your custom-trained CNN model.")

# ----------- Text input -----------
st.header("📝 Analyze a Single Sentence")
text_input = st.text_area("Enter Arabic text:")

if st.button("Analyze Text"):
    if text_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        tokens = clean_and_tokenize(text_input)
        encoded = encode(tokens, vocab).unsqueeze(0)
        with torch.no_grad():
            output = model(encoded)
            pred = torch.argmax(output, dim=1).item()
        st.success(f"🧠 Sentiment: **{idx2label[pred]}**")

# ----------- CSV input -----------
st.header("📂 Analyze a CSV File")
uploaded_file = st.file_uploader("Upload a CSV with `ID` and `Commentaire` columns", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if "ID" not in df.columns or "Commentaire" not in df.columns:
            st.error("CSV must contain columns: `ID` and `Commentaire`.")
        else:
            df['tokens'] = df['Commentaire'].astype(str).apply(clean_and_tokenize)
            df['encoded'] = df['tokens'].apply(lambda x: encode(x, vocab))
            inputs = torch.stack(df['encoded'].tolist())

            with torch.no_grad():
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1).numpy()
            df['Statut'] = [idx2label[i] for i in preds]

            # Pie Chart
            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = df['Statut'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')
            st.pyplot(fig)

            st.subheader("📄 Preview of Predictions")
            st.dataframe(df[['ID', 'Commentaire', 'Statut']])

            # Download result
            csv_output = df[['ID', 'Commentaire', 'Statut']].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", data=csv_output, file_name="sentiment_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
