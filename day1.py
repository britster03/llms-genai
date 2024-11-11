import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from chromadb import Client
import torch

load_dotenv()

def main():
    st.header("Chat with PDF 💬")
    
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    if pdf:
        pdf_reader = PdfReader(pdf)
        text = "".join([page.extract_text() for page in pdf_reader.pages])
        st.success("PDF uploaded and text extracted!")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text)
        
        embeddings = HuggingFaceEmbeddings()
        chroma_client = Client()
        collection = chroma_client.create_collection("pdf_embeddings")
        collection.add(documents=chunks, embeddings=embeddings.embed_documents(chunks))
        
        st.success("Text embedded and stored in ChromaDB!")
        query = st.text_area("Ask a question about your PDF:")
        
        if query:
            docs = collection.query(query_texts=[query], n_results=5)["documents"][0]
            llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature": 0.2, "max_length": 1000})
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write("**Answer:**", response)
            
            collection.add(documents=[response], embeddings=embeddings.embed_documents([response]))
            st.subheader("Feedback:")
            st.write("Was this answer helpful?")
            thumbs_up, thumbs_down = st.columns(2)
    
            if thumbs_up.button("👍"):
                st.success("Thanks for your feedback!")
            elif thumbs_down.button("👎"):
                st.info("Generating an alternative answer...")
                llm_alt = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.3, "max_length": 1000})
                chain_alt = load_qa_chain(llm=llm_alt, chain_type="stuff")
                response_alt = chain_alt.run(input_documents=docs, question=query)
                st.write("**Alternative Answer:**", response_alt)
