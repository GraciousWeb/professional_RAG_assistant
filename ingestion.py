import os
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv(override=True)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")


def make_id(doc):
    """Generate a unique ID for a document based on its metadata and content."""
    base = f'{doc.metadata.get("source","")}|{doc.metadata.get("page","")}|{doc.page_content}'
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def ingest_docs():
    """Takes a pdf, converts it to embeddings and uploads it to Pinecone"""
    if not pinecone_index_name:
        raise ValueError("Missing PINECONE_INDEX_NAME in environment variables.")
    if not openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment variables.")

    print("Loading PDF...")
    loader = PyPDFLoader("data/iso27001.pdf")
    raw_document = loader.load() #converts the pdf into a list of Document objects which contains the text and metadata of the pdf

    print("Splitting document...")
    text_splitter = RecursiveCharacterTextSplitter(
     chunk_size=800,
     chunk_overlap=150, #without overlap, important texts can be split in half
     separators=["\n\n", "\n", ".", " "]) #paragraphs, line breaks sentences and spaces respectively

    documents = text_splitter.split_documents(raw_document)

    #loop through each document chunk and add metadata (source and page number) to track where data came from
    for d in documents:
        d.metadata["source"] = d.metadata.get("source", "iso27001.pdf") #if source is not found, use the default value
        d.metadata["page"] = d.metadata.get("page", None)

    ids = [make_id(d) for d in documents]
    print(f"Created {len(documents)} chunks.")

    print("Powering up Embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("Uploading to Pinecone...")

    PineconeVectorStore.from_documents(documents, embeddings, index_name=pinecone_index_name)
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_docs()
