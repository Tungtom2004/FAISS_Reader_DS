import os 
from dotenv import load_dotenv 
from PyPDF2 import PdfReader
from langchain_openai import AzureChatOpenAI 
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

#read pdf
load_dotenv()

def raw_pdf(s):
    raw_text = ''
    for i,page in enumerate(s.pages):
        text = page.extract_text()
        if text:
            raw_text += text 
    return raw_text


doc_reader = PdfReader('D:/ChatbotQA_DS/Introduction-to-Data-Science.pdf')
raw_text = raw_pdf(doc_reader)

#chunking 
splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size = 450, 
    chunk_overlap = 60,
    length_function = len
)

texts = splitter.split_text(raw_text)

#embeddings
embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

db = FAISS.from_texts(texts, embeddings)
db.save_local("D:/ChatbotQA_DS/vectorstore")

print("Completed successfully!")

