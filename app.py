import streamlit as st
import pickle
import re

st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Enter a news article to check whether it is REAL or FAKE")

# ✅ Load saved model + vectorizer
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# Input
st.subheader("Enter News Text")
user_input = st.text_area("Paste full news article here", height=200)

# Prediction
if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        # Clean text
        cleaned = re.sub(r'\W', ' ', user_input.lower())

        # Transform
        vectorized = vectorizer.transform([cleaned])

        # Predict
        prediction = model.predict(vectorized)
        prob = model.predict_proba(vectorized)

        if prediction[0] == 1:
            st.success(f"✅ REAL News ({round(prob[0][1]*100, 2)}%)")
        else:
            st.error(f"❌ FAKE News ({round(prob[0][0]*100, 2)}%)")