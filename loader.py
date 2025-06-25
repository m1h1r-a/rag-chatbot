import langchain
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "/home/m1h1r/Documents/[2] dev/tvs-chatbot/data"


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


docs = load_documents()
chunks = split_docs(docs)

print(len(docs))
print("\n\n\n\n")
print(docs[2])
print("\n\n\n\n")
print(len(chunks))
print(chunks[2])
