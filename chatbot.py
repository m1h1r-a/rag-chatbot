import os
import shutil

import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from loader import check_and_load_documents

DATA_PATH = "/home/m1h1r/Documents/[2] dev/tvs-chatbot/data"
CHROMA_PATH = "/home/m1h1r/Documents/[2] dev/tvs-chatbot/chroma"


PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---
Answer the question based on the above context, use markdown and emojis in your reply : {question}
"""


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


def query_rag(question: str):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    results = db.similarity_search_with_score(question, k=5)
    context = "\n---\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=question)
    # print(prompt)

    model = OllamaLLM(model="llama3:8b")
    response = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    output = f"\n\n{response}\n\nSources:\n\n{sources}"
    print(output)
    return output


# question = input("You:\n")
# query_rag(question)

st.title("Ask The Chatbot")

col1, col2 = st.columns(2)


@st.dialog("Info")
def show_dialog(data):
    st.write(data)


with col1:
    if st.button("Check for new Documents"):
        data = check_and_load_documents()
        show_dialog(data)

with col2:
    if st.button("Clear Database"):
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            show_dialog("Database Cleared")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("What's your Question?")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    answer = query_rag(prompt)
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
