from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import argparse
import os
import shutil


CHROMA_PATH= "chroma"
DATA_PATH = "data/"

def add_to_chroma(chunks: list[Document]):
    #load db
    db = Chroma(
        persist_directory=CHROMA_PATH,embedding_function=get_embeddings()
    )

    #calculating page id
    chunks_with_ids = calculate_chunk_ids(chunks)

     # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")


    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f" Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print(" No new documents to add")
    
    
   # data = db.get()
   # document_ids = data["ids"]

    #for doc_id in zip(document_ids):
    #    print(f"ID: {doc_id}")

    

def calculate_chunk_ids(chunks):

    # source,pagenumber, chunk number
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        #if page id and is same as last one, increment the index
        if current_page_id == last_page_id:
            current_chunk_index +=1
        else:
            current_chunk_index = 0
    
        #calc chunkid 
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks






def get_embeddings():
    embeddings = OllamaEmbeddings( model= "nomic-embed-text" )
    return embeddings


def load_docs():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap =500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    document = chunks[1]
    print(document.page_content)
    print(document.metadata)

    return chunks

def generate_data_store():
    documents = load_docs()
    chunks = split_text(documents)
    add_to_chroma(chunks)
    
 
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("clearing db")
        clear_database()

    generate_data_store()



if __name__ == "__main__":
    main()