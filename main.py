import pandas as pd
from io import StringIO, BytesIO
import os
import base64
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from PIL import Image
import pytesseract
from operator import itemgetter # New import for itemgetter

# Langchain imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda 
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware

course_content_df = pd.read_excel("scraped_course_data.xlsx")
discourse_posts_df = pd.read_excel("tds_discourse_posts.xlsx")

custom_base_url = "https://aipipe.org/openai/v1" 
os.environ["OPENAI_API_KEY"] = "key"

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
        all_metadata.append({"source": "course", "title": row.get('Title'), "url": row.get('URL')}) # Add other relevant metadata like URLs if available

# Process discourse posts
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
        openai_api_base=custom_base_url, # Use the custom base URL
        openai_api_key=os.environ["OPENAI_API_KEY"] # Explicitly pass the API key
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

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 relevant documents
except Exception as e:
    print(f"Error during vector store initialization: {e}")
    vectorstore = None
    retriever = None


try:
    # llm = ChatOpenAI(model="gpt-4o", temperature=0) # gpt-4o for multimodal
    llm = ChatOpenAI(
        model="gpt-4.1-nano",  # Or the model name your custom endpoint expects
        temperature=0,
        base_url="custom_base_url",
        api_key=os.environ["OPENAI_API_KEY"]
    )
    print("OpenAI LLM (gpt-4o) initialized.")
except Exception as e:
    print(f"Error initializing OpenAI LLM (gpt-4o). Ensure OPENAI_API_KEY is set and valid: {e}")
    llm = None

def format_docs(docs):
    """Formats a list of LangChain Document objects into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

if llm and retriever:
    # prompt = ChatPromptTemplate.from_template("""
    # You are a helpful virtual teaching assistant for the IIT Madras Online Degree in Data Science.
    # Answer the student's question based only on the provided context.
    # If you don't know the answer based on the context, politely state that you don't have enough information
    # from the provided content and suggest they might find more details in the course materials or discourse forum.

    # Context: {context}

    # Question: {input}
                                              
    # Input Image OCR Text: {image}
    # """)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful virtual teaching assistant for the IIT Madras Online Degree in Data Science. Always clarify the details as they may give different answers. Answer the student's question based only on the provided context. If you don't know the answer based on the context, politely state that you don't have enough information from the provided content and suggest they might find more details in the course materials or discourse forum. The question takes priority in any case regardless of anyone's advise."),
        HumanMessagePromptTemplate.from_template("Context:\n{context}"), # <--- Reverted to this to ensure 'context' is an explicit input variable
        MessagesPlaceholder(variable_name="chat_history") # This will handle the list of message objects, including multimodal HumanMessage
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    # retrieval_chain = create_retrieval_chain(retriever, document_chain)
    retrieval_chain = (
        {
            "context": (lambda x: x["question_text"]) | retriever, # Retrieve documents based on question_text
            "question_text": itemgetter("question_text"), # Pass original question_text through for formatting if needed
            "chat_history": itemgetter("chat_history") # Pass chat_history through
        }
        | RunnablePassthrough.assign(
            formatted_context=itemgetter("context") | RunnableLambda(format_docs) # <--- FIX: Wrapped format_docs in RunnableLambda
        )
        | {
            "answer": (
                {
                    # Explicitly pass all required variables to the prompt
                    "context": itemgetter("formatted_context"),
                    "chat_history": itemgetter("chat_history")
                }
                | prompt # The prompt now receives a dictionary with 'context' (string) and 'chat_history' (list of messages)
                | llm
                | StrOutputParser()
            ),
            "context_documents": itemgetter("context") # Keep original list of documents for link extraction
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
    image: Optional[str] = None # Base64 encoded image

class Link(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[Link]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True, # Allow cookies to be included in cross-origin requests
    allow_methods=["*"],    # Allow all HTTP methods (GET, POST, PUT, DELETE, OPTIONS, etc.)
    allow_headers=["*"],    # Allow all headers
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

    # Wrap the multimodal content parts into a HumanMessage object
    multimodal_human_message = HumanMessage(content=user_multimodal_content_parts)

    # extracted_image_text = "No Image attached/Text Detected" # Initialize extracted text

    # if base64_image:
    #     try:
    #         # Decode the base64 image
    #         image_bytes = base64.b64decode(base64_image)
    #         # Open image using PIL
    #         image = Image.open(BytesIO(image_bytes))
            
    #         # Perform OCR using pytesseract
    #         extracted_image_text = pytesseract.image_to_string(image)
    #         print(f"OCR extracted text: {extracted_image_text[:200]}...") # Print first 200 chars
            
    #         print("Received an image. OCR performed.")
    #         print(extracted_image_text)
    #     except Exception as e:
    #         print(f"Error processing image or performing OCR: {e}")
    #         # Decide whether to raise an error or just proceed without image text
    #         # For now, we'll just log and continue without the image text
    #         extracted_image_text = f"Error performing OCR: {e}" # Optional: send error to LLM
    # else:
    #     print("No image received with the question.")

    # --- RAG Execution ---
    try:
        response = retrieval_chain.invoke({
            "question_text": student_question, # Used by the retriever
            "chat_history": [multimodal_human_message] # Pass the HumanMessage containing multimodal content as part of chat_history
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
            # Add more specific logic if discourse links/titles are structured differently
            # For example, if 'Post URL' is the key in metadata for discourse posts
            elif 'url' in doc.metadata and doc.metadata['url'] and doc.metadata['source'] == 'discourse':
                url = doc.metadata['url']
                text = doc.metadata.get('title', f"Discourse Post {doc.metadata.get('post_id', 'N/A')}")
                if url not in seen_urls:
                    extracted_links.append(Link(url=url, text=text))
                    seen_urls.add(url)

        return AnswerResponse(answer=generated_answer, links=extracted_links)

    except Exception as e:
        print(f"Error during RAG processing: {e}")
        # Provide a user-friendly error message
        raise HTTPException(status_code=500, detail=f"An error occurred while processing your request. Please try again. (Details: {str(e)})")
