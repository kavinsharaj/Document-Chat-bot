from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


def create_vector_store(pdf_path: str, persist_path: str = "vector_store"):
    """
    Loads a PDF, splits it into text chunks, creates embeddings, and saves them in a Chroma vector store.

    Args:
        pdf_path (str): Path to the PDF file.
        persist_path (str): Directory to save the vector store. Default is "vector_store".
    """
    loader = UnstructuredPDFLoader(pdf_path)
    documents = loader.load()
    for i, doc in enumerate(documents):
        doc.metadata["source"] = pdf_path
        doc.metadata["page"] = i + 1
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(documents)
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        chunks, embedding, persist_directory=persist_path
    )
    vectorstore.persist()
    print(f"Vector store saved at '{persist_path}' with {len(chunks)} chunks.")
