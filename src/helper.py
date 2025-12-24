from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import Config


# Function: Load the pdf files from "data" directory
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)

    documents = loader.load()
    return documents


# Function: Filter the Documents to minimal information
def filter_to_minimal_docs(docs: list[Document]) -> list[Document]:
    """
    input: The list of Document
    output: The list of minimal Documents containing (src,page_content)
    """

    minimal_docs: list[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(page_content=doc.page_content, metadata={"source": src})
        )
    return minimal_docs


# Function: Perfrom Text Splitting on Documents
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


# Function: Download embedding model 
def download_embeddings():
    """
    Downlaod and return the HuggingFace embeddings model.
    """
    model_name = Config.EMBEDDINGS_MODEL
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
