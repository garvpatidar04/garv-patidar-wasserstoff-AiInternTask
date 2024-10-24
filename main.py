from SummaryKeywordUtils import process_single_pdf
from db_connection import logger
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
import psutil
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from timeit import default_timer as time

load_dotenv('.env')
pdf_path = os.getenv('pdf_path')

def summary_keyword_extract(file_name_list):
    """Extracts text, summarizes, and extracts keywords for multiple PDFs sequentially."""
    
    # Logging the total number of files to process
    logger.info(f"Processing {len(file_name_list)} PDF files.")
    
    for file_name in file_name_list:
        try:
            # Process each PDF file sequentially
            result = process_single_pdf(file_name)
            # You can log or handle the result as needed
            logger.info(f"Successfully processed {file_name}.")
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")


def main():
    try:
        start_process = time()
        pdf_file_list = [os.path.join(pdf_path, f) for f in os.listdir(pdf_path) if f.endswith('.pdf')]
        # summary_keyword_extract_concurrently(pdf_file_list)
        summary_keyword_extract(pdf_file_list)
        end_process = time()
        logger.info(f"Total time taken: {end_process-start_process}")
    except Exception as e:
        logger.error(f"Error in Processing {str(e)}")
        raise


if __name__=='__main__':
    main()