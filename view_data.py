import streamlit as st
import pandas as pd
import re

# --- CLEANING FUNCTION ---
def clean(text):
    text = re.sub(r"http\S+|@\S+|#\S+", "", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- LOADERS ---
@st.cache_data
def load_isot():
    df_fake = pd.read_csv("data/ISOT/Fake.csv")
    df_real = pd.read_csv("data/ISOT/True.csv")
    df_fake['label'] = 'fake'
    df_real['label'] = 'real'
    df_fake['source'] = 'ISOT'
    df_real['source'] = 'ISOT'
    df_fake['text'] = (df_fake['title'].fillna('') + ". " + df_fake['text']).map(clean)
    df_real['text'] = (df_real['title'].fillna('') + ". " + df_real['text']).map(clean)
    return df_fake[['text', 'label', 'source']], df_real[['text', 'label', 'source']]

@st.cache_data
def load_welfake():
    df = pd.read_csv("data/WELFake.csv")
    df['label'] = df['label'].map({0: 'real', 1: 'fake'})
    df['source'] = 'WELFake'
    df['text'] = df['text'].map(clean)
    return df[['text', 'label', 'source']]

# --- MAIN APP ---
def view_data():
    st.set_page_config(page_title="Fake News Dataset Viewer", layout="wide")
    st.title("🧠 Fake News Dataset Viewer")

    df_fake_isot, df_real_isot = load_isot()
    df_welfake = load_welfake()

    # Combine all
    combined_df = pd.concat([df_fake_isot, df_real_isot, df_welfake], ignore_index=True)

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Dataset Filters")
    dataset_choice = st.sidebar.selectbox("Select View", ["All Combined", "ISOT - Fake", "ISOT - Real", "WELFake", "Fake Only Combined"])
    sample_count = st.sidebar.slider("How many samples to view?", min_value=10, max_value=1000, value=20, step=10)

    # --- DATA SELECTION ---
    if dataset_choice == "ISOT - Fake":
        data = df_fake_isot
    elif dataset_choice == "ISOT - Real":
        data = df_real_isot
    elif dataset_choice == "WELFake":
        data = df_welfake
    elif dataset_choice == "Fake Only Combined":
        data = combined_df[combined_df['label'] == 'fake']
    else:
        data = combined_df

    st.markdown(f"### Showing: {dataset_choice} ({len(data)} total records)")
    st.dataframe(data.sample(n=min(sample_count, len(data))), use_container_width=True)

    # --- EXPORT FOR GENERATOR ---
    if st.button("📁 Export Fake News for GPT-2 Generator"):
        fake_only = combined_df[combined_df['label'] == 'fake']
        fake_text = "\n".join(fake_only['text'].tolist())
        st.download_button("Download fake_news_combined.txt", fake_text, file_name="fake_news_combined.txt", mime="text/plain")

    # --- EXPORT FOR DETECTOR ---
    if st.button("📁 Export Data for DistilBERT Detector"):
        export_df = combined_df[['text', 'label']].copy()
        export_df['label'] = export_df['label'].map({'real': 0, 'fake': 1})
        csv_data = export_df.to_csv(index=False)
        st.download_button("Download detector_training_data.csv", csv_data, file_name="detector_training_data.csv", mime="text/csv")

# --- RUN APP ---
if __name__ == "__main__":
    view_data()
