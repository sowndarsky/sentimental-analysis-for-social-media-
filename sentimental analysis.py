from dotenv import load_dotenv
from langchain_google_genai.llms import GoogleGenerativeAI
import streamlit as st
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# ✅ Set custom tab name and icon
st.set_page_config(
    page_title="Social Media Sentiment Analyzer",  # Browser tab title
    page_icon="💬",                                # Tab icon (emoji or image URL)
    layout="centered"
)

st.title("💬 Social Media Sentiment Analyzer")
st.info("Paste text or upload a CSV file to analyze sentiment.")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = st.sidebar.text_input("AIzaSyDjC1F4bss3DEOxkIRhX3hRU-u5dwjx3yI", type="password", value=api_key)

# Function to get LLM sentiment response
def get_response(post_text):
    try:
        llm = GoogleGenerativeAI(
            api_key=GEMINI_API_KEY,
            model="gemini-2.0-flash"
        )
        prompt = f"""
Analyze the sentiment of the following social media post.

Post: "{post_text}"

Respond with:
- Sentiment: Positive, Negative, or Neutral
- Explanation: Brief reason for the sentiment classification
- Suggestion: (Optional) How the tone could be improved if needed
"""
        return llm.invoke(prompt)
    except Exception as e:
        return f"An error occurred: {e}"

# UI
st.title("💬 Social Media Sentiment Analyzer")
st.info("Paste text or upload a CSV file to analyze sentiment.")

# Option 1: Manual Text Input
post_text = st.text_area("📲 Enter a single social media post:")
single_submit = st.button("🔍 Analyze One Post")

# Option 2: CSV Upload
csv_file = st.file_uploader("📄 Or upload CSV file with a 'post_text' column", type="csv")
bulk_submit = st.button("📊 Analyze CSV Dataset")

# Process single post
if single_submit:
    if not GEMINI_API_KEY:
        st.warning("Please enter your Gemini API key.")
    elif not post_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            response = get_response(post_text)
            st.write(response)
            st.success("✅ Sentiment analysis complete!")

# Process dataset
if bulk_submit:
    if not GEMINI_API_KEY:
        st.warning("Please enter your Gemini API key.")
    elif csv_file is None:
        st.warning("Please upload a CSV file.")
    else:
        df = pd.read_csv(csv_file)
        if "post_text" not in df.columns:
            st.error("CSV must have a 'post_text' column.")
        else:
            st.info(f"Analyzing {len(df)} posts...")
            results = []
            for post in df["post_text"]:
                result = get_response(post)
                results.append(result)
            df["sentiment_result"] = results
            st.success("✅ All posts analyzed!")
            st.dataframe(df)
            st.download_button("⬇️ Download Results as CSV", data=df.to_csv(index=False), file_name="sentiment_results.csv")
