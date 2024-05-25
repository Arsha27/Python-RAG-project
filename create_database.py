from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings



DATA_PATH = "data/"

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
 


def main():
    generate_data_store()

if __name__ == "__main__":
    main()