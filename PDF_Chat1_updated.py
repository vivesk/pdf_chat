import os
import streamlit as st
from PyPDF2 import PdfReader
# langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import re

# api key config
os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY'
# define LLM
llm_gemini = ChatGoogleGenerativeAI(model="gemini-pro")
# define vector store file name
file_name = 'all_vec_db'

## Function to clean the extracted text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces/newlines
    text = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', text)  # Remove special characters
    return text

## Function to load and clean PDFs
def pdf_loader(files):
    all_text = ''
    for file in files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            all_text += clean_text(page.extract_text())
    return all_text


## function for chunking
def chunk_creator(all_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    chunks = text_splitter.split_text(all_text)
    return chunks


## function for creating embeddings
def embedding_creator(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Setting up vector db
    vecstore = FAISS.from_texts(chunks, embeddings)
    vecstore.save_local(file_name)


## function for retrieving similar vectors from db
def retrieve_similar(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.load_local(file_name, embeddings, allow_dangerous_deserialization=True)
    vector_db.index.nprobe = 5  # Adjust n_probe based on performance
    similar_texts = vector_db.similarity_search(query)
    return similar_texts


## function for getting answer from the LLM
def retrieve_answer_LLM(docs,query):
    p_template = '''
    Based on the provided context, answer the following question as thoroughly and accurately as possible. 
    Ensure that your answer is detailed and well-explained. 
    If the information needed to answer the question is not present in the context, respond with "The answer is not available in the context."
    Avoid providing incorrect information or information that isnt present in context.
    Please structure your answer in three detailed paragraphs. \n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    '''
    prompt = PromptTemplate(template=p_template, input_variables=['context','question'])
    chain = load_qa_chain(llm_gemini, chain_type='stuff', prompt=prompt)
    response = chain({"input_documents":docs,"question": query},
                        return_only_outputs=True)
    
    return response["output_text"]


# Page config
st.header("PDF QnA BOT🤖")
st.sidebar.title("Add PDFs")
uploaded_files = st.sidebar.file_uploader("Browse", accept_multiple_files=True)
upload_button = st.sidebar.button("Upload")
sidebar_placeholder = st.sidebar.empty()
st.subheader("Enter Query:")    
user_query = st.text_input("Talk to the files!")
submit_button = st.button("Submit")


# action on file upload
if upload_button:
    sidebar_placeholder.text("Reading Files...")
    all_pdfs_text = pdf_loader(uploaded_files)
    sidebar_placeholder.text("Chunking Text...")
    text_chunked = chunk_creator(all_pdfs_text)
    sidebar_placeholder.text("Creating Text Embeddings...")
    embedding_creator(text_chunked)
    sidebar_placeholder.text("Files read successfully!")



# action on query submission
if submit_button:
    pbar = st.progress(25,text='Retrieving similar Chunks...')
    docs = retrieve_similar(user_query)
    pbar.progress(50,"Querying LLM...")
    result = retrieve_answer_LLM(docs, user_query)
    pbar.progress(100,"Displaying Results")
    st.subheader("Results:")
    st.write(result)
