import streamlit as st
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests
import os

# ==================== YOUTUBE VIDEO MAPPING ====================
YOUTUBE_VIDEOS = {
    "1": "https://youtu.be/yH1zCq-iaeU",   # What is SQL
    "2": "https://youtu.be/HmH-76_2Ak8",   # Data types, Primary Key
    "3": "https://youtu.be/v-2cIUgx_jw",   # Create database, Create table
    "4": "https://youtu.be/4YAAgrm8_ZI",   # Insert, Update, Delete
    "5": "https://youtu.be/eiLqDeDp7Oc",   # Select, Where
    "6": "https://youtu.be/rfWYbMd3ApA",   # Import Excel
    "7": "https://youtu.be/55_UN5gVARs",   # String functions
    "8": "https://youtu.be/9NfthspfXEo",   # Aggregate functions
    "9": "https://youtu.be/SvJLXj05cow",   # Group By
    "10": "https://youtu.be/kwGh6XvLrEk",  # Timestamp & Extract
    "11": "https://youtu.be/H6988OpZKTU",  # SQL Joins
}

# ==================== YOUTUBE TIMESTAMP LINK ====================
def generate_youtube_timestamp_link(video_number, start_seconds):
    if str(video_number) not in YOUTUBE_VIDEOS:
        return None
    base_url = YOUTUBE_VIDEOS[str(video_number)]
    return f"{base_url}?t={int(start_seconds)}s"

# ==================== UTILS ====================
def format_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

@st.cache_resource
def load_embeddings():
    try:
        return joblib.load("embeddings.joblib")
    except FileNotFoundError:
        st.error("❌ embeddings.joblib not found")
        return None

def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "bge-m3", "input": text_list},
        timeout=30
    )
    return r.json()["embeddings"]

def inference(prompt, model="llama3.2"):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120
    )
    return r.json()

def find_relevant_content(df, question, top_k):
    q_emb = create_embedding([question])[0]
    sims = cosine_similarity(np.vstack(df["embedding"]), [q_emb]).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    result = df.loc[top_idx].copy()
    result["similarity"] = sims[top_idx]
    return result

# ==================== STREAMLIT APP ====================
def main():
    st.set_page_config(page_title="RAG SQL Assistant", layout="wide")
    st.title("🤖 RAG based SQL Assistant")
    st.caption("Ask questions about your SQL course videos")

    df = load_embeddings()
    if df is None:
        return

    # ===== SIDEBAR (CLEANED) =====
    with st.sidebar:
        st.header("⚙️ Settings")
        model_choice = st.selectbox("Select Model", ["llama3.2", "deepseek-r1"])
        top_k = st.slider("Chunks to retrieve", 3, 10, 5)
        show_raw = st.checkbox("Show raw data", False)

        st.markdown("---")
        st.info(f"Total chunks: {len(df)}")

    # ===== MAIN UI =====
    question = st.text_input("💬 Ask your question", "How do I use SQL JOIN?")
    if st.button("🔍 Ask"):
        with st.spinner("Searching..."):
            relevant_df = find_relevant_content(df, question, top_k)

        prompt = f"""
Here are video chunks:
{relevant_df[['title','number','start','end','text']].to_json(orient='records')}

Question: {question}
Answer clearly and mention video number and timestamp.
"""
        response = inference(prompt, model_choice)["response"]

        st.markdown("### 💡 Answer")
        st.markdown(response)

        st.markdown("---")
        st.markdown("### 🎬 Watch on YouTube")

        for video_num in relevant_df["number"].unique():
            chunks = relevant_df[relevant_df["number"] == video_num]
            st.subheader(f"📺 Video {video_num}: {chunks.iloc[0]['title']}")

            for _, row in chunks.iterrows():
                link = generate_youtube_timestamp_link(row["number"], row["start"])
                st.markdown(
                    f"[▶ Watch at {format_timestamp(row['start'])}]({link}) — {row['text'][:80]}..."
                )

        if show_raw:
            st.markdown("---")
            st.dataframe(relevant_df)

if __name__ == "__main__":
    main()
