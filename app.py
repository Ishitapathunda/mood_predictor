import streamlit as st
from transformers import pipeline
import pandas as pd
from datetime import datetime
import base64

# --- App Config ---
st.set_page_config(page_title="Mood Predictor", layout="centered")

# --- Custom CSS for Dark Theme ---
st.markdown("""
    <style>
    body {
        background-color: #111111;
        color: white;
    }
    .stTextArea textarea {
        background-color: #1e1e1e !important;
        color: white !important;
    }
    .stButton button {
        background-color: #444444;
        color: white;
    }
    .stDownloadButton button {
        background-color: #555555;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load CPU-compatible model ---
classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=True
)

# --- Mood emoji map ---
emoji_map = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

# --- App Header ---
st.markdown("## 📝 Mood Predictor from Journal Entry")
st.markdown("Enter your journal entry to analyze your mood.")

# --- Input ---
user_input = st.text_area("Write your journal entry here:")

# --- Session state for history ---
if "history" not in st.session_state:
    st.session_state["history"] = []

# --- Predict Button ---
if st.button("Predict Mood") and user_input.strip() != "":
    result = classifier(user_input)[0]
    top_mood = max(result, key=lambda x: x["score"])
    mood_label = top_mood["label"]
    emoji = emoji_map.get(mood_label, "")
    score = round(top_mood["score"] * 100, 2)

    st.success(f"**Predicted Mood:** {mood_label.capitalize()} {emoji} ({score}%)")

    # Save to history
    st.session_state["history"].append({
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Mood": mood_label,
        "Score": score,
        "Text": user_input
    })

# --- Mood History ---
if st.session_state["history"]:
    st.markdown("### 📈 Mood History")
    df = pd.DataFrame(st.session_state["history"])
    st.dataframe(df)

    # --- Download CSV ---
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📁 Download Mood History as CSV", csv, "mood_history.csv", "text/csv")
