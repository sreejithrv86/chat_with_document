import os

from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_groq import ChatGroq
from logging_config import logger

# Load environment variables
load_dotenv()
key = os.getenv('GROQ_API_KEY')
model = os.getenv('MODEL')

class Chain:
    def __init__(self):
        # initialize llm model in the in Groq
        self.llm = ChatGroq(
            model_name=model,
            temperature=0,
            groq_api_key=key
        )
        logger.info(f"llm is initialized")

    def create_qa_chain(self, retriever):
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        logger.info(f"Successfully generated qa_chain!")
        return qa_chain
