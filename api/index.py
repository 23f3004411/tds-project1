import pandas as pd
import os
from typing import Optional, List

from fastapi import FastAPI, HTTPException, File, Form
from pydantic import BaseModel, Field
from PIL import Image
from operator import itemgetter

# Langchain imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware

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
    print("Embedding model loaded.")

    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded successfully.")
    else:
        print("FAISS index not found. Creating a new one...")
        if all_texts:
            vectorstore = FAISS.from_texts(all_texts, embeddings, metadatas=all_metadata)
            vectorstore.save_local(FAISS_INDEX_PATH)
            print(f"FAISS index created and saved to {FAISS_INDEX_PATH}.")
        else:
            vectorstore = None
            print("No text data available to create FAISS index.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
except Exception as e:
    print(f"Error during vector store initialization: {e}")
    vectorstore = None
    retriever = None

try:
    llm = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0,
        base_url=custom_base_url,
        api_key=openai_api_key # Use the variable here
    )
    print("OpenAI LLM (gpt-4.1-nano) using aipipe.org initialized.")
except Exception as e:
    print(f"Error initializing OpenAI LLM (gpt-4.1-nano). Ensure OPENAI_API_KEY is set and valid: {e}")
    llm = None

def format_docs(docs):
    """Formats a list of LangChain Document objects into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

if llm and retriever:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
                    You are a helpful virtual teaching assistant for the IIT Madras Online Degree in Data Science.
                    Always clarify the details as they may give different answers.
                    Answer the student's question based only on the provided context.
                    If you don't know the answer based on the context, politely state that you don't have enough information from the provided content
                    and suggest they might find more details in the course materials or discourse forum.
                    """),
        HumanMessagePromptTemplate.from_template("Context:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = (
        {
            "context": (lambda x: x["question_text"]) | retriever,
            "question_text": itemgetter("question_text"),
            "chat_history": itemgetter("chat_history")
        }
        | RunnablePassthrough.assign(
            formatted_context=itemgetter("context") | RunnableLambda(format_docs)
        )
        | {
            "answer": (
                {
                    "context": itemgetter("formatted_context"),
                    "chat_history": itemgetter("chat_history")
                }
                | prompt
                | llm
                | StrOutputParser()
            ),
            "context_documents": itemgetter("context")
        }
    )
    print("RAG chain created.")
    print("RAG chain created with multimodal support for gpt-4o.")
else:
    print("RAG chain could not be created due to missing LLM or Retriever.")
    retrieval_chain = None


app = FastAPI(
    title="TDS Virtual TA API",
    description="API for IIT Madras TDS Virtual Teaching Assistant using Local Models",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[Link]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/")
async def read_root():
    return {"message": "Hello World!"}

@app.options("/api/")
async def api_options():
    return {"message": "OPTIONS request allowed at /api/"}

@app.post("/api/", response_model=AnswerResponse)
async def get_answer(request_data: QuestionRequest):
    if not retrieval_chain:
        raise HTTPException(status_code=500, detail="RAG system not initialized. Check server logs for LLM/Embeddings/FAISS errors.")

    student_question = request_data.question
    base64_image = request_data.image

    user_multimodal_content_parts = [{"type": "text", "text": student_question}]
    if base64_image:
        user_multimodal_content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        })
        print("Received an image. gpt-4o will attempt to interpret it along with the text.")
    else:
        print("No image received with the question.")

    multimodal_human_message = HumanMessage(content=user_multimodal_content_parts)
    try:
        response = retrieval_chain.invoke({
            "question_text": student_question,
            "chat_history": [multimodal_human_message]
        })

        generated_answer = response["answer"]
        context_documents = response["context_documents"]

        extracted_links = []
        seen_urls = set()

        for doc in context_documents:
            if 'url' in doc.metadata and doc.metadata['url'] and 'title' in doc.metadata:
                url = doc.metadata['url']
                text = doc.metadata['title']
                if url not in seen_urls:
                    extracted_links.append(Link(url=url, text=text))
                    seen_urls.add(url)
            elif 'url' in doc.metadata and doc.metadata['url'] and doc.metadata['source'] == 'discourse':
                url = doc.metadata['url']
                text = doc.metadata.get('title', f"Discourse Post {doc.metadata.get('post_id', 'N/A')}")
                if url not in seen_urls:
                    extracted_links.append(Link(url=url, text=text))
                    seen_urls.add(url)

        return AnswerResponse(answer=generated_answer, links=extracted_links)

    except Exception as e:
        print(f"Error during RAG processing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing your request. Please try again. (Details: {str(e)})")