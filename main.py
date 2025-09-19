import os 
from dotenv import load_dotenv 
from PyPDF2 import PdfReader
from langchain_openai import AzureChatOpenAI 
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import re 

vs_dir = "D:/ChatbotQA_DS/vectorstore"

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs = {'device':'cpu'},
    encode_kwargs = {'normalize_embeddings':True}
)

llm = AzureChatOpenAI(
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version = os.getenv("AZURE_OPENAI_VERSION"),
    api_key = os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0
)


db = FAISS.load_local(vs_dir,embeddings,allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity",search_kwargs = {"k":6})

qa_chain = RetrievalQA.from_chain_type(
    llm = llm, 
    retriever = retriever, 
    chain_type = "stuff",
    return_source_documents = True
)


def clean_text(text):
    text = re.sub(r'\\\[|\\\]', '', text)
    text = re.sub(r'\\\(|\\\)', '', text)
    return text

#Query - Answer
if __name__=="__main__":
    q = input("Enter your problem:")
    res = qa_chain.invoke({"query":q})
    answer = clean_text(res['result'])
    print(f"Answer: {answer}")
