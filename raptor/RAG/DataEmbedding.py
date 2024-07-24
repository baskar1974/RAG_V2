from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
import os 
import PyPDF2
import torch

def load_chunk_persist_pdf():
    pdf_folder_path = "demo/calix_pdf"
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
                                                chunk_size=1000,
                                                chunk_overlap=40,
                                                length_function=len,
                                                is_separator_regex=False,)
    chunked_documents = text_splitter.split_documents(documents)
    model_name = "intfloat/e5-base-v2"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    embeddings = HuggingFaceBgeEmbeddings(
                                            model_name=model_name,
                                            model_kwargs={'device': 'cuda'},  #getting RuntimeError if used 'device':'cuda' due to compatibility issues
                                            encode_kwargs=encode_kwargs
                                            )
    vector_db = FAISS.from_documents(chunked_documents,embedding=embeddings)
    vector_db.save_local("faiss_index")
    new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_kwargs={"k": 4})   #retrieving top 4 similar documents  
    return retriever