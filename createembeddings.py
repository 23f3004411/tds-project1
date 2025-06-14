import pandas as pd

course_content_df = pd.read_excel("scraped_course_data.xlsx")
discourse_posts_df = pd.read_excel("tds_discourse_posts.xlsx")

# Example using Langchain's RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Experiment with this value
    chunk_overlap=200, # Experiment with this value
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

all_texts = []
all_metadata = []

# Process course content
for index, row in course_content_df.iterrows():
    text = str(row['Main Article Content'])
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        all_texts.append(chunk)
        all_metadata.append({"source": "course", "title": row.get('Title', f"Course Content Part {i+1}"), "url": row.get('URL')}) # Add other relevant metadata like URLs if available

# Process discourse posts
for index, row in discourse_posts_df.iterrows():
    text = str(row['Content'])
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        all_texts.append(chunk)
        all_metadata.append({"source": "discourse", "post_id": row.get('Post ID'), "url": row.get('Post URL')})

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(all_texts, embeddings, metadatas=all_metadata)
vectorstore.save_local("faiss_index")