# app.py
import streamlit as st
from detector import predict_fake_news
from generator import generate_fake_news, calculate_perplexity

st.set_page_config(page_title="Fake News Generator & Detector", layout="wide")
st.title("📰 Fake News Generator & Detector")

# Sidebar
mode = st.sidebar.radio("Select Mode", ["Fake News Detector", "Fake News Generator"])

# Detector Mode
if mode == "Fake News Detector":
    st.header("🕵️ Detect Fake News")

    news_input = st.text_area("Enter News Article or Headline", height=200)

    if st.button("🔍 Compare Detection"):
        if news_input.strip() == "":
            st.warning("Please enter some news text.")
        else:
            # Pretrained prediction
            pre_label, pre_conf, _ = predict_fake_news(news_input, use_finetuned=False)

            # Fine-tuned prediction
            fine_label, fine_conf, _ = predict_fake_news(news_input, use_finetuned=True)

            # Show side-by-side comparison
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🧪 Pretrained Model")
                st.markdown(f"**Prediction:** {pre_label}")
                st.markdown(f"**Confidence:** {pre_conf}%")

            with col2:
                st.markdown("### ✅ Fine-Tuned Model")
                st.markdown(f"**Prediction:** {fine_label}")
                st.markdown(f"**Confidence:** {fine_conf}%")

# Generator Mode
elif mode == "Fake News Generator":
    st.header("🧠 Generate Fake News")
    #topic = st.selectbox("Select Topic", ["Politics", "Technology", "Health", "Other"])
    prompt = st.text_area("Enter a Prompt (e.g., 'A new virus has emerged...')", height=150)

    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("Temperature", 0.5, 1.5, 1.0)
    with col2:
        length = st.slider("Max Length", 50, 250, 100)

    if st.button("🚀 Generate"):
        if prompt.strip() == "":
            st.warning("Please enter a prompt.")
        else:
            # Generate from both models
            fine_news = generate_fake_news(prompt, max_length=length, temperature=temp, use_finetuned=True)
            fine_perplexity = calculate_perplexity(fine_news, use_finetuned=True)

            pre_news = generate_fake_news(prompt, max_length=length, temperature=temp, use_finetuned=False)
            pre_perplexity = calculate_perplexity(pre_news, use_finetuned=False)

            # Show results side-by-side
            st.subheader("📄 Generated Fake News Comparison")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### ✅ Fine-Tuned Model Output")
                st.info(fine_news)
                st.markdown(f"🔢 Perplexity: `{fine_perplexity}`")

            with col2:
                st.markdown("#### 🧪 Pretrained Model Output")
                st.info(pre_news)
                st.markdown(f"🔢 Perplexity: `{pre_perplexity}`")
