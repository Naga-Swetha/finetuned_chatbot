import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Cohere 
COHERE_API_KEY = "KpKTNDvVLORJrBLuuktrNPHbMmzVNSdaJFikUOcz"
st.header("Fine-Tuning implementation Chatbot with Cohere API")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file", type="pdf")
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    user_query = st.text_input("Ask a question about your document:")
    if user_query:
        match = vector_store.similarity_search(user_query)
        llm = Cohere(
            cohere_api_key=COHERE_API_KEY,
            model="command", 
            temperature=0.5,
            max_tokens=700
        )
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_query)
        st.write(response)
    else:
        st.write("Please enter a question to get started.")
else:
    st.write("Please upload a PDF file to start.")

