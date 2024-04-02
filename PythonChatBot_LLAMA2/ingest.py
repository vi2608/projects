from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.text_splitter import RecursiveCharacterTextSplitter

DATA_P = 'Data/'
DB_P = 'vectorstore/db_faiss'

def create_db():
    loader = DirectoryLoader(DATA_P,
    glob='*.pdf',
    loader_cls=PyPDFLoader)

    documents = loader.load()
    text_split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_split.split_documents(documents)

    hf = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, hf)
    db.save_local(DB_P)

if __name__ == "__main__":
   create_db()