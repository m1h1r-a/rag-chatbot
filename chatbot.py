import os
import shutil

import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from loader import check_and_load_documents

DATA_PATH = "data"
CHROMA_PATH = "chroma"


PROMPT_TEMPLATE = """
Answer the question based only on the following context, if there is no context or you are unsure, 
do not try making up an answer, do not rely on general knowledge and just let me know that there is no context,
by writing it out nicely and asking the user to try again with a different question:
{context}

---
Answer the question based on the above context, use markdown, and do not use emojis in your reply : {question}
"""


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


def query_rag(question: str, selected_sources=None):

    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    if selected_sources:
        full_paths = [os.path.join(DATA_PATH, source) for source in selected_sources]
        results = db.similarity_search_with_score(
            question,
            k=5,
            filter={"source": {"$in": full_paths}},
        )
    else:
        results = db.similarity_search_with_score(question, k=5)

    context = "\n---\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=question)
    # print(prompt)

    model = OllamaLLM(model="llama3:8b")
    response = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    # output = f"\n\n{response}\n\nSources:\n\n{sources}"
    output = f"\n\n{response}\n\n"
    print(output)
    return output


# -----------------------------------------------------------STREAMLIT-------------------------------------------------


st.title("Ask The Chatbot")
st.set_page_config(page_title="Chatbot", page_icon="🤖")

if "selected_source" not in st.session_state:
    st.session_state.selected_source = None

sources = os.listdir(DATA_PATH)
st.sidebar.title("Select a Document")

if st.sidebar.button("Home"):
    st.session_state.selected_source = None

for src in sources:
    name = src.split(".")[0]
    if st.sidebar.button(f"Chat with {name}"):
        st.session_state.selected_source = src


if st.session_state["selected_source"]:

    src = st.session_state["selected_source"]
    name = src.split(".")[0]
    st.write(f"### Chatbot For {name}")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("What's your Question?")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        answer = query_rag(prompt, [st.session_state["selected_source"]])
        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:

    col1, col2, col3 = st.columns(3)

    @st.dialog("Info")
    def show_dialog(data):
        st.write(data)

    with col1:
        selected_sources = st.multiselect(
            "Sources", sources, placeholder="Sources", label_visibility="collapsed"
        )

    with col2:
        if st.button("Check for new Documents"):
            data = check_and_load_documents()
            show_dialog(data)

    with col3:
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
        answer = query_rag(prompt, selected_sources)
        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
