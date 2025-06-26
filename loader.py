import langchain
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core import embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "/home/m1h1r/Documents/[2] dev/tvs-chatbot/data"
CHROMA_PATH = "/home/m1h1r/Documents/[2] dev/tvs-chatbot/chroma"


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


def calculate_chunk_ids(chunks):
    chunk_index = 0
    prev_id = None
    for chunk in chunks:

        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        cur_id = f"{source}:{page}"
        # print(cur_id, "\n")

        if cur_id == prev_id or prev_id == None:
            final_id = f"{cur_id}:{chunk_index}"
            print(f"{final_id}\n")
            chunk_index += 1
            prev_id = cur_id
        else:
            chunk_index = 0
            final_id = f"{cur_id}:{chunk_index}"
            print(f"{final_id}\n")
            chunk_index += 1
            prev_id = cur_id
        chunk.metadata["id"] = final_id
    return chunks


docs = load_documents()
chunks = split_docs(docs)
chunks = calculate_chunk_ids(chunks)
