import streamlit as st
from dotenv import load_dotenv
import os

from chain import Chain
from chroma_db import ChromaDb
from read_pdf import ReadPDF
from logging_config import logger

# Set page config at the very beginning
st.set_page_config(layout="centered",
                   page_title="LLAMA Chatbot 3.1",
                   page_icon="🦙")

# Load environment variables
load_dotenv()
lang = os.getenv('LANGUAGE')

# Cache or use session state to avoid re-initialization
if "chromadb" not in st.session_state:
    logger.info("Initializing ChromaDb for the first time...")
    st.session_state.chromadb = ChromaDb()

if "chain" not in st.session_state:
    logger.info("Initializing Chain for the first time...")
    st.session_state.chain = Chain()

if "readpdf" not in st.session_state:
    logger.info("Initializing ReadPDF for the first time...")
    st.session_state.readpdf = ReadPDF()

chromadb = st.session_state.chromadb
chain = st.session_state.chain
readpdf = st.session_state.readpdf


def create_streamlit_app(chain, chromadb):
    # Use a form to handle both the Enter key press and button click
    with st.form(key="query_form", clear_on_submit=False):
        # Input field for URL
        query_input = st.text_input("Enter Your Query:", value="What are the corrections made as part of Version 1.3?")
        logger.info("query = " + query_input)

        # Submit button
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            if query_input is not None and query_input.strip() != "":
                logger.info("Button clicked or Enter pressed. Processing query...")
                retriever = chromadb.retrive_chroma_collection()
                logger.info("Retriever created from ChromaDb.")

                qa_chain = chain.create_qa_chain(retriever)
                logger.info("QA chain created.")

                response = chromadb.query_qa_chain(qa_chain, query_input)
                logger.info("Query processed. Displaying response.")

                # Add custom CSS for a grey box with visible scroll bar and proper text wrapping
                st.markdown(
                    """
                    <style>
                    .grey-box {
                        background-color: #f0f0f0;  /* Light grey background */
                        padding: 10px;              /* Padding inside the box */
                        border-radius: 5px;         /* Optional: rounded corners */
                        white-space: normal;        /* Allow text to wrap normally */
                        word-wrap: break-word;      /* Break long words if necessary */
                        border: 1px solid #ccc;     /* Optional: border color */
                        color: black;               /* Set font color to black */
                        max-height: 400px;          /* Set max height for the box */
                        overflow-y: scroll;         /* Always show vertical scroll bar */
                        list-style-position: inside; /* Ensure bullet points are aligned correctly */
                    }

                    /* Scrollbar Styling */
                    .grey-box::-webkit-scrollbar {
                        width: 12px;               /* Set width of the scrollbar */
                    }

                    .grey-box::-webkit-scrollbar-track {
                        background: #f1f1f1;       /* Background of the scrollbar track */
                    }

                    .grey-box::-webkit-scrollbar-thumb {
                        background-color: #888;    /* Color of the scrollbar handle */
                        border-radius: 10px;       /* Optional: Rounded corners for the handle */
                        border: 2px solid #f1f1f1; /* Optional: Add a border around the handle */
                    }

                    .grey-box::-webkit-scrollbar-thumb:hover {
                        background-color: #555;    /* Change handle color when hovered */
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Display the response inside the grey box
                st.markdown(f"<div class='grey-box'>{response}</div>", unsafe_allow_html=True)
                logger.info("Response displayed.")
            else:
                logger.info("Button clicked or Enter pressed, but the query is blank...")
                st.markdown(
                    """
                    <style>
                    .grey-box {
                        background-color: #f0f0f0;  /* Light grey background */
                        padding: 10px;              /* Padding inside the box */
                        border-radius: 5px;         /* Optional: rounded corners */
                        white-space: normal;        /* Allow text to wrap normally */
                        word-wrap: break-word;      /* Break long words if necessary */
                        border: 1px solid #ccc;     /* Optional: border color */
                        color: black;               /* Set font color to black */
                        max-height: 400px;          /* Set max height for the box */
                        overflow-y: scroll;         /* Always show vertical scroll bar */
                        list-style-position: inside; /* Ensure bullet points are aligned correctly */
                    }

                    /* Scrollbar Styling */
                    .grey-box::-webkit-scrollbar {
                        width: 12px;               /* Set width of the scrollbar */
                    }

                    .grey-box::-webkit-scrollbar-track {
                        background: #f1f1f1;       /* Background of the scrollbar track */
                    }

                    .grey-box::-webkit-scrollbar-thumb {
                        background-color: #888;    /* Color of the scrollbar handle */
                        border-radius: 10px;       /* Optional: Rounded corners for the handle */
                        border: 2px solid #f1f1f1; /* Optional: Add a border around the handle */
                    }

                    .grey-box::-webkit-scrollbar-thumb:hover {
                        background-color: #555;    /* Change handle color when hovered */
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Display the response inside the grey box
                st.markdown(
                    f"<div class='grey-box'>You haven't asked a question yet. Please go ahead and ask your question, and I'll do my best to answer it based on the provided context (which appears to be empty).</div>",
                    unsafe_allow_html=True)
                logger.info("No Response due to no query.")


if __name__ == "__main__":
    logger.info("Set page config for Streamlit")
    st.title("🦙 LLAMA Chatbot 3.1")

    # Process PDF once and store the result in session state
    if "documents" not in st.session_state:
        logger.info("Reading and processing PDF for the first time...")
        texts = readpdf.convert_pdf_text_into_chunks()
        documents = readpdf.convert_chunks_into_vector_embeddings(texts)
        st.session_state.documents = documents
    else:
        documents = st.session_state.documents

    # Populate ChromaDb only once and ensure proper write status
    if "write_status" not in st.session_state:
        write_status = chromadb.populate_chroma_collection(documents)
        st.session_state.write_status = write_status
    else:
        write_status = st.session_state.write_status

    if write_status:
        logger.info(f"Successfully wrote the contents from PDF to ChromaDB!")
    else:
        logger.error(f"Failed to write the contents from PDF to ChromaDB!")
        st.session_state.chat_active = False
        st.write("Chat has been closed.")

    # Create Streamlit app UI for querying
    create_streamlit_app(chain, chromadb)