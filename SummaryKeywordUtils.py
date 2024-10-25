from db_connection import logger, collection
import warnings
warnings.filterwarnings("ignore")
import os
import fitz  
import re 
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
from string import punctuation
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import spacy
nlp = spacy.load("en_core_web_sm")

def get_pdf_size_category(file_path):
    """Categorize PDF based on number of pages"""
    try:
        doc = fitz.open(file_path)
        page_count = len(doc)
        doc.close()

        if page_count <= 30:
            return 'short', 5
        elif page_count <= 50:
            return 'medium', 8
        elif page_count <= 100:
            return 'long', 12
        else:
            return 'too_long', 15

    except Exception as e:
        logger.info(f"Error checking PDF size for {file_path}: {str(e)}")
        return None

def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ''
        for page in doc:
            text += page.get_text() 
        md={}
        md['pdf_size'] = f"{round(os.path.getsize(file_path)/(1024))} KB"
        md['pdf_path'] = file_path
        md['document_name'] = file_path.split('/')[-1]
        md_u = doc.metadata
        md_u.update(md)
        doc.close()

        return text, md_u

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def clean_text(text)-> str:
    # regex pattern for unwanted character and text 
    unwanted_pattern = r"[Ââ1]|[^\w\s]|(?<!\w)[a-hj-zA-HJ-Z](?!\w)"
    hindi_word = r"[ँ-ःअ-ऍए-ऑओ-नप-रलळव-ह़-ॅे-ॉो-्ॐ]"
    page_num = r"\d+\s+of\s+\d+"
    result = ''
    clean_content = re.sub(unwanted_pattern, '', text)
    clean_content = re.sub(hindi_word, '', clean_content)
    clean_content = re.sub(r'\s+', ' ', clean_content)
    clean_content = re.sub(page_num, '', clean_content)
    result = clean_content.strip()
    text_size = len(result.split(' '))

    return result, text_size


def load_model():
    checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32)
    return tokenizer, base_model


def split_text(text, max_length=512):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


def llm_pipeline(sum_size):
    tokenizer,base_model = load_model()
    min_l = min(50, sum_size-30)
    return pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=sum_size,
        min_length=min_l,
        do_sample=False
    )


# Function to summarize text chunks using the pipeline
def summarize_text(text_chunks):
    sum_size = min(100, min([len(_.split(' ')) for _ in text_chunks]))
    pipe_sum = llm_pipeline(sum_size)
    summaries = []
    for chunk in text_chunks:
        result = pipe_sum(chunk)
        summary = result[0]['summary_text']
        summaries.append(summary)
    return summaries


def get_hotwords(text):
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN'] 
    doc = nlp(text.lower()) 
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
    return result


def add_metadata(metadata):
    doc = {
        'filename': metadata['document_name'],
        'metadata': metadata
    }
    try:
        collection.insert_one(doc)
        logger.info('Added metadata to the database')
    except Exception as e:
        logger.error(f'Error Pushing to database: {str(e)}')

def update_db(file_name, update_data):
    try:
        collection.update_one(
                {"filename": file_name},
                update_data
            )
        logger.info('Updated database for summary and keywords')
    except Exception as e:
        logger.error(f'Error updating database: {str(e)} for summary and keywords {file_name}')
        raise


def process_single_pdf(file_path):
    """Process a single PDF with appropriate number of workers"""
    try:
        _, num_workers = get_pdf_size_category(file_path)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future = executor.submit(extract_text_from_pdf, file_path)
            # This will return the pdf text and pdf metadata (text, metadata)
            pdf_text, pdf_metadata = future.result()
            add_metadata(pdf_metadata)
            logger.info(f"Text is extracted from {pdf_metadata['document_name']}.")

        # Clean the text for summerisation and keyword extraction
        cleaned_text, text_size = clean_text(pdf_text)
        logger.info(f"Text cleaning is done of {pdf_metadata['document_name']}.")

        # summerize the text
        logger.info(f"Summerizing {pdf_metadata['document_name']}.")
        text_chunks = split_text(cleaned_text)
        summary = summarize_text(text_chunks)
        logger.info(f"Summerization of {pdf_metadata['document_name']} is done.")

        #keyword extraction
        logger.info(f"Extracting Keywords from {pdf_metadata['document_name']}.")
        output = get_hotwords(cleaned_text)
        most_common_list = Counter(output).most_common(30)
        keywords = [key for key,_ in most_common_list]
        logger.info(f"Extracting Keywords from {pdf_metadata['document_name']} is completed.")

        update_data = {
            "$set": {
                "summary": summary,
                "keywords": keywords
            }
        }

        update_db(
            pdf_metadata['document_name'],
            update_data
        )
        log.info("Updated the database with summary and keywords")
    
    except Exception as e:
        logger.error(f"Some unknown error happened in the pipeline {e}")
        raise e
