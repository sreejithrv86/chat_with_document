import logging
import logging.config
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
log_file_paths = os.getenv('LOGFILE_PATH')

# Define log file path
log_file_path = os.path.join(log_file_paths,
                             'chat_with_document' + '_' + datetime.now().strftime("%Y-%m-%d %H_%M_%S") + '.log')

# Configure logging
logging.basicConfig(
    filename=log_file_path,
    filemode='a',  # Append to the log file
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    level=logging.INFO  # Set the logging level to DEBUG for detailed logs
)

# Create a logger
logger = logging.getLogger(__name__)