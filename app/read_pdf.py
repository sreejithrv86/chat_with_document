import os
import warnings

from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from transformers import AutoTokenizer
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
from logging_config import logger

# Suppress specific warning message
warnings.filterwarnings("ignore",
                        message="You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.")

# Suppress specific FutureWarning from Hugging Face Transformers
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Load environment variables
load_dotenv()
file_path = os.getenv('PDF_FILE_PATH')
file_name = os.getenv('FILE_NAME')

class ReadPDF:

    def __init__(self):
        # Get the current working directory
        self.current_dir = os.getcwd()
        self.pdf_file_path = os.path.join(self.current_dir, file_path, file_name)
        self.loader = UnstructuredLoader(self.pdf_file_path)
        self.document = self.loader.load()

    def convert_pdf_text_into_chunks(self):
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Assuming `documents` is a list of Document objects to process
        text_splitter = CharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400
        )
        # Split documents into manageable chunks
        texts = text_splitter.split_documents(self.document)
        logger.info(f"convert pdf text into chunks task is successful!")
        return texts

    def convert_chunks_into_vector_embeddings(self, texts):
        # Accessing the text content for tokenization
        pages_content = [doc.page_content for doc in texts]

        # Tokenize the texts
        texts_tokenized = [self.tokenizer.encode(text, add_special_tokens=True) for text in pages_content]

        # Decode the tokenized texts (if needed)
        texts_decoded = [
            self.tokenizer.decode(
                tokenized_text,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False  # Explicitly set to False
            )
            for tokenized_text in texts_tokenized
        ]
        # Create Document objects for Chroma from decoded texts
        documents = [Document(page_content=text) for text in texts_decoded]
        logger.info(f"convert chunks into vector embeddings task is successful!")
        return documents



