
streamlit_code = """
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

st.title("🤖 AI Candidate Checker")

@st.cache_resource
def load_model_and_tokenizer():
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)

    class TextCNN(nn.Module):
        def __init__(self, vocab_size, embed_size, num_classes, max_len):
            super(TextCNN, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.conv1 = nn.Conv2d(1, 100, (3, embed_size))
            self.conv2 = nn.Conv2d(1, 100, (4, embed_size))
            self.conv3 = nn.Conv2d(1, 100, (5, embed_size))
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(3 * 100, num_classes)

        def conv_and_pool(self, x, conv):
            x = torch.relu(conv(x)).squeeze(3)
            x = torch.max_pool1d(x, x.size(2)).squeeze(2)
            return x

        def forward(self, x):
            x = self.embedding(x)
            x = x.unsqueeze(1)
            x1 = self.conv_and_pool(x, self.conv1)
            x2 = self.conv_and_pool(x, self.conv2)
            x3 = self.conv_and_pool(x, self.conv3)
            x = torch.cat((x1, x2, x3), 1)
            x = self.dropout(x)
            return self.fc1(x)

    model = TextCNN(vocab_size=10000, embed_size=128, num_classes=2, max_len=200)
    model.load_state_dict(torch.load("textcnn_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

transcript_input = st.text_area("📝 Enter Interview Transcript:")
job_role_input = st.text_input("💼 Enter Job Role:")
csv_file = st.file_uploader("📁 Or upload CSV file with transcripts", type=["csv"])

def clean_text(text):
    return re.sub(r'\s+', ' ', str(text)).strip()

def preprocess_and_predict(text):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    input_tensor = torch.tensor(padded, dtype=torch.long)
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    return "✅ Fit" if predicted_class == 1 else "❌ Not Fit"

if st.button("Predict from Text"):
    if transcript_input.strip() == "":
        st.warning("Please enter a transcript.")
    else:
        result = preprocess_and_predict(transcript_input)
        st.subheader(f"Prediction for Job Role: {job_role_input or 'N/A'}")
        st.success(result)

if csv_file:
    df = pd.read_csv(csv_file)
    if "Transcript" not in df.columns:
        st.error("CSV must have a 'Transcript' column.")
    else:
        df["Cleaned"] = df["Transcript"].apply(clean_text)
        sequences = tokenizer.texts_to_sequences(df["Cleaned"])
        padded = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
        input_tensor = torch.tensor(padded, dtype=torch.long)
        with torch.no_grad():
            predictions = model(input_tensor)
        df["Prediction"] = torch.argmax(predictions, dim=1).numpy()
        df["Prediction"] = df["Prediction"].map({0: "Not Fit", 1: "Fit"})
        st.subheader("📊 Predictions on Uploaded CSV")
        st.dataframe(df[["Transcript", "Prediction"]])
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
"""
