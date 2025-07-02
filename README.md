# TVS Chatbot

This project is a chatbot designed to answer questions about TVS motorcycle models. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide answers based on a collection of PDF documents. The chatbot is built with Streamlit, LangChain, and a local Large Language Model (LLM).

## Features

- **Chat with Specific Documents:** Select a specific PDF document to chat with.
- **Chat with Multiple Documents:** Select multiple PDF documents to chat with.
- **Document Loading:** Automatically loads and processes new PDF documents from the `data` directory.
- **Vector Database:** Uses ChromaDB to store and retrieve document embeddings.
- **Local LLM:** Utilizes a local LLM for generating answers.

## Project Structure

```
.
├── chatbot.py           # Main Streamlit application
├── loader.py            # Handles document loading and processing
├── requirements.txt     # Project dependencies
├── data/                # Directory for PDF documents
│   ├── iqube.pdf
│   ├── jupiter.pdf
│   └── rtr.pdf
└── chroma/              # Directory for ChromaDB data
```

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/tvs-chatbot.git
    cd tvs-chatbot
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up the local LLM:**

    This project uses a local LLM. Make sure you have a compatible LLM running and accessible.

## Usage

1.  **Add PDF documents:**

    Place your PDF documents in the `data` directory.

2.  **Run the Streamlit application:**

    ```bash
    streamlit run chatbot.py
    ```

3.  **Interact with the chatbot:**

    - Open your browser to the Streamlit URL.
    - Use the sidebar to select a specific document to chat with, or use the main interface to chat with multiple documents.
    - Click "Check for new Documents" to load any new PDFs into the database.
    - Click "Clear Database" to remove all documents from the database.