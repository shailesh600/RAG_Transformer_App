import os
import requests
import streamlit as st

# ✅ Auto-switch between local and Render API URL
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/query/")

# Streamlit UI
st.title("🔍 RAG Transformer Issue Finder")

query = st.text_input("Enter a transformer issue description:")

if st.button("Search"):
    if query:
        try:
            response = requests.post(API_URL, json={"query": query})
            if response.status_code == 200:
                results = response.json().get("retrieved_documents", [])
                st.write("### 🔎 Related Transformer Issues:")
                for idx, result in enumerate(results, start=1):
                    st.write(f"**{idx}.** {result}")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
