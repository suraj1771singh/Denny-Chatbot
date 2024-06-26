from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from embeddings import get_embeddings
import shutil
import os
import argparse


DATA_PATH = "data"
CHROMA_PATH = "db"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()
    else:
        docouments = load_documents()
        chunks = split_documents(docouments)
        add_to_chroma(chunks)


"""
    # Create the parser
    parser = argparse.ArgumentParser(description="A script that can reset the database or process a document.")
    
    # Create a mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)
    
    # Add the --reset argument to the group
    group.add_argument("--reset", action="store_true", help="Reset the database.")
    
    # Add the document path argument to the group
    group.add_argument("--document-path", type=str, help="Path to the document.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Handle the arguments
    if args.reset:
        print("✨ Clearing Database")
        clear_database()
    elif args.document_path:
        process_document(args.document_path)
    else:
        parser.print_help()
"""


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    embedding = get_embeddings()

    # Load the persisted database from disk
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

    # Calculate Page IDs
    chunks_with_ids = cal_chunk_id(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    # Add only documents that don't exist in DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print("Adding new documents.")
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunks_ids)
        print("Succesfully added new documents.")

    else:
        print("No new documents to add.")


def cal_chunk_id(chunks):
    prev_page_id = None
    curr_chunk_ind = 0
    for chunk in chunks:
        src = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        curr_page_id = f"{src}:{page}"

        if curr_page_id == prev_page_id:
            curr_chunk_ind += 1
        else:
            curr_chunk_ind = 0

        chunk_id = f"{curr_page_id}:{curr_chunk_ind}"
        prev_page_id = curr_page_id
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
