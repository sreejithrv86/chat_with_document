import os
import shutil
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from logging_config import logger

# Load environment variables
load_dotenv()

collection_name = os.getenv('CHROMA_COLLECTION_NAME', 'default_collection_name')  # Default if not set
model_name = os.getenv('EMBEDDING_MODEL', 'default_model_name')
chroma_telemetry = os.getenv('CHROMA_TELEMETRY_ENABLED', 'false')
batch_size = int(os.getenv('BATCH_SIZE', 10))  # Default batch size


class ChromaDb:

    def __init__(self):
        # Convert the texts into vector embeddings using HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # Disable telemetry for Chroma
        os.environ["CHROMA_TELEMETRY_ENABLED"] = chroma_telemetry

        # Path to the collection (usually stored in a directory)
        self.collection_path = os.path.join('app', collection_name)
        #self.collection_path = os.path.join(collection_name)

        # Check if the collection exists (directory exists)
        if os.path.exists(self.collection_path):
            logger.info(f"Collection '{collection_name}' exists. Attempting to delete...")
            try:
                shutil.rmtree(self.collection_path)
                logger.info(f"Successfully deleted the collection '{collection_name}'")
            except PermissionError as e:
                logger.error(f"PermissionError: {e}. Retrying after a delay...")
                time.sleep(1)  # Delay before retrying
                try:
                    shutil.rmtree(self.collection_path)
                    logger.info(f"Successfully deleted the collection '{collection_name}' on retry.")
                except Exception as e:
                    logger.error(f"Failed to delete the collection after retry: {e}")
                    raise

        # Initialize Chroma with the embedding function
        self.vectordb = Chroma(embedding_function=self.embeddings, persist_directory=self.collection_path)
        logger.info(f"ChromaDB initialized with collection '{collection_name}'")


    def populate_chroma_collection(self, documents):
        # Add documents to Chroma in batches
        for batch in self.batch_data(documents, batch_size):
            self.vectordb.add_documents(batch)
        logger.info("Successfully inserted documents into ChromaDB!")
        return True

    def retrive_chroma_collection(self):
        retriever = self.vectordb.as_retriever()
        logger.info("Successfully created the retriever and is returned!")
        return retriever

    # Function to split data into smaller batches
    def batch_data(self, data, batch_size):
        # Ensure `data` is a list-like object
        if not isinstance(data, list):
            raise TypeError("Expected `data` to be a list of documents.")

        # Yield batches of documents
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def query_qa_chain(self, qa_chain, query):
        # Invoke QA chain and get a response to user query
        response = qa_chain.invoke({"query": query})
        logger.info("The response is generated successfully!")
        return response["result"]


if __name__ == "__main__":
    # Test the environment variables and ensure values are loaded correctly
    print(f"Collection Name: {collection_name}")