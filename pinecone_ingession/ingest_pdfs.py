
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.schema import Document
from langchain_pinecone import PineconeVectorStore

from src.config import Config
from src.helper import download_embeddings

DATA_DIR = "data"



def ingest_pdfs():
    print("📄 Loading all PDFs in data folder...")
    loader = DirectoryLoader(DATA_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    print(f"✅ Loaded {len(docs)} pages from all PDFs")

    # Keep minimal metadata (important for clean RAG)
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )

    # Medical-friendly chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(minimal_docs)

    print(f"✂️ Split into {len(chunks)} chunks")

    # Embeddings
    print("🔤 Loading embedding model...")
    embeddings = download_embeddings()

    # Pinecone
    print("🌲 Connecting to Pinecone...")
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=Config.PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    # Upload
    print("🚀 Uploading vectors to Pinecone...")
    vectorstore.add_documents(chunks)

    print("🎉 Ingestion completed successfully!")


if __name__ == "__main__":
    ingest_pdfs()
