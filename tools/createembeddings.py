

import pandas as pd
import os

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

course_content_df = pd.read_excel("scraped_data/scraped_course_data.xlsx")
discourse_posts_df = pd.read_excel("scraped_data/tds_discourse_posts.xlsx")

# Retrieve custom_base_url from environment variable
custom_base_url = os.environ.get("OPENAI_API_BASE_URL", "https://aipipe.org/openai/v1")

# Retrieve API key from environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# You don't need to set it again here if it's already in os.environ
# os.environ["OPENAI_API_KEY"] = "YOUR_KEY_HERE" # REMOVE THIS LINE

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

all_texts = []
all_metadata = []

for index, row in course_content_df.iterrows():
    text = str(row['Main Article Content'])
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        all_texts.append(chunk)
        all_metadata.append({"source": "course", "title": row.get('Title'), "url": row.get('URL')})

for index, row in discourse_posts_df.iterrows():
    text = str(row['Content'])
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        all_texts.append(chunk)
        all_metadata.append({"source": "discourse", "post_id": row.get('Post ID'), "url": row.get('Post URL')})

FAISS_INDEX_PATH = "faiss_index"

try:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_base=custom_base_url,
        openai_api_key=openai_api_key # Use the variable here
    )
    vectorstore = FAISS.from_texts(all_texts, embeddings, metadatas=all_metadata)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index created and saved to {FAISS_INDEX_PATH}.")
except Exception as e:
    print(f"Error during vector store initialization: {e}")
    vectorstore = None
    retriever = None