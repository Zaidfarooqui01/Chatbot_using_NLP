
import os
import json
from datetime import datetime
import csv
import ssl
import nltk
import streamlit as st # type: ignore
import random
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# Base directory
BASE_DIR = Path(__file__).parent

# Disable SSL verification for nltk data downloads
ssl.create_default_context = ssl._create_unverified_context
# Ensure punkt is available (download once to local nltk_data)
nltk.data.path.append(str(BASE_DIR / "nltk_data"))
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt", download_dir=str(BASE_DIR / "nltk_data"))

# Load intents
intents_path = BASE_DIR / "intents.json"
with open(intents_path, "r", encoding="utf-8") as f:
    intents = json.load(f)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def encode_patterns(model, sentences):
    return model.encode(sentences, convert_to_numpy=True)

# Prepare training data
sentences = []
tags = []
for intent in intents:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        tags.append(intent["tag"])

# Load model and encode with spinner
with st.spinner("Loading model and preparing embeddings..."):
    model = load_model()
    embeddings = encode_patterns(model, sentences)

# Ensure unrecognized log exists with header
unrec_path = BASE_DIR / "unrecognized_log.csv"
if not unrec_path.exists() or unrec_path.stat().st_size == 0:
    with open(unrec_path, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["input_text", "timestamp"])

# Streamlit UI
st.set_page_config(page_title="Smart Chatbot", layout="centered")
st.title("💬 Smart Chatbot with NLP")
st.markdown("Ask me anything!")

# Similarity threshold configurable
SIMILARITY_THRESHOLD = st.sidebar.slider("Similarity threshold", 0.4, 0.9, 0.6, 0.01)

def chatbot(input_text: str) -> str:
    input_vec = model.encode([input_text], convert_to_numpy=True)[0]
    cos_scores = util.cos_sim(input_vec, embeddings)[0]
    # convert to numpy if needed
    if hasattr(cos_scores, "cpu"):
        cos_scores = cos_scores.cpu().numpy()
    elif hasattr(cos_scores, "numpy"):
        cos_scores = cos_scores.numpy()

    best_idx = int(np.argmax(cos_scores))
    best_score = float(cos_scores[best_idx])
    best_tag = tags[best_idx]

    if best_score < SIMILARITY_THRESHOLD:
        with open(unrec_path, "a", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            csv_writer.writerow([input_text, timestamp])

        for intent in intents:
            if intent.get("tag") == "fallback":
                return random.choice(intent.get("responses", ["I'm not sure I understand. Can you rephrase?"]))
        return "I'm not sure I understand. Can you rephrase?"

    for intent in intents:
        if intent.get("tag") == best_tag:
            return random.choice(intent.get("responses", ["Sorry, something went wrong."]))

    return "Sorry, something went wrong."

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your message here...")

if user_input:
    response = chatbot(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display chat history
for sender, message in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(message)
