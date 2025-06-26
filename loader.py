from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_ollama import OllamaEmbeddings
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
            # print(f"{final_id}\n")
            chunk_index += 1
            prev_id = cur_id

        else:
            chunk_index = 0
            final_id = f"{cur_id}:{chunk_index}"
            # print(f"{final_id}\n")
            chunk_index += 1
            prev_id = cur_id
        chunk.metadata["id"] = final_id
    return chunks


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    chunk_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    existing = f"Existing Items in DB : {len(existing_ids)}"

    new = []
    for chunk in chunk_ids:
        if chunk.metadata["id"] not in existing_ids:
            new.append(chunk)

    if len(new) != 0:
        new_chunks_added = f"{len(new)} New chunks Added"
        new_chunk_ids = [chunk.metadata["id"] for chunk in new]
        db.add_documents(new, ids=new_chunk_ids)
        total = f"Total Items in DB : {len(existing_ids)}"
        output = f"{existing}\n\n{new_chunks_added}\n\n{total}"
    else:
        print("No New Documents to add")
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        total = f"Total Items in DB : {len(existing_ids)}"
        output = f"{existing}\n\nNo New Documents to Add\n\n{total}"

    return output


def check_and_load_documents():
    docs = load_documents()
    chunks = split_docs(docs)
    data = add_to_chroma(chunks)
    return data
