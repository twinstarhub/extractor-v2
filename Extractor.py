import jieba
import re
import torch
import json
from common import save_list_to_file 
from langdetect import detect
from tqdm import tqdm
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, util

# Chinese - English Extracter

class Extractor :
    def __init__(self,en_filepath_in, cn_filepath_in,en_filepath_out ,cn_filepath_out):
        self.cn_filepath_in = cn_filepath_in
        self.en_filepath_in = en_filepath_in
        self.cn_filepath_out = cn_filepath_out
        self.en_filepath_out = en_filepath_out

        # Internal variables
        self.cn_length = 0              # Number of extracted chainese sentences
        self.en_length = 0              # Number of extracted chainese sentences
        self.cn_linecount = 0           # Number of letters in one sentences chinese(average length)
        self.en_linecount = 0           # Number of letters in one sentences english(average length)
        
        # Result variables
        self.cn_sentences_list = []
        self.en_sentences_list = []

        self.extracting_en_time = 0.0   # English extracted time
        self.extracting_cn_time = 0.0   # Chinese extracted time
        self.extracting_cn_time = 0.0   # Pairiing time

        self.accuracy = 0.0             # Result paired paragraph accuracy  result/total percentage
        self.limit_score = 0.8      # Limit score of similarity 
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu', cache_folder='./')


    # Extracting Chinese Basic
    def extract_chinese_basic(self):
        #  This function is basic functiong extracting chinese documents to text without processing them
        cn_text = extract_text(self.cn_filepath_in)
        cn_paragraphs = cn_text.split('\n\n')
        with open(self.cn_filepath_out, "w", encoding="utf-8") as file:
            file.write(cn_text)
        return cn_paragraphs
    
    # Extracting English Basic
    def extract_english_basic(self):
        #  This function is basic functiong extracting english documents to text without processing them
        en_text = extract_text(self.en_filepath_in)

        en_paragraphs = en_text.split('\n\n')
        with open(self.en_filepath_out, "w", encoding="utf-8") as file:
            file.write(en_text)
        return en_paragraphs
    

    def is_chinese(self,sentence):
        for char in sentence:
            if len(jieba.lcut(char)):
                return True
            return False
    

    def extract_chinese_second(self):
        print('--> Generating paragraph...')
        result = []
        basic_paragraph_list = self.extract_chinese_basic()
        total_paragraphs = len(basic_paragraph_list)
        with tqdm(total=total_paragraphs, desc='Processing paragraphs') as pbar:
            for paragraph in basic_paragraph_list:
                # proceed_paragraph = self.chinese_segment_process(paragraph)
                if self.is_chinese(paragraph):
                    result.append(paragraph)
                else :
                    pass
                pbar.update(1)
        return result
    
    # Necessary word checking for symbols and numbers
    def contains_necessary_words(self,text):

        try:
            lang = detect(text)
            return lang == 'en' or re.search(r'[\u4e00-\u9fff]', text) is not None
        except:
            return False
    
    # Test process extracting
    def extract_paragraphs_from_pdf(self,pdf_path):
        paragraphs = []
        current_paragraph = ''

        text = extract_text(pdf_path)
        lines = text.split('\n')

        with tqdm(total=len(lines), desc="Extracting paragraphs") as pbar:
            for line in lines:
                if line.strip():  # Skip empty lines
                    current_paragraph += line.strip() + ' '
                elif current_paragraph:  # Reached the end of a paragraph
                    if self.contains_necessary_words(current_paragraph):
                        paragraphs.append(current_paragraph.strip())
                    current_paragraph = ''

                pbar.update(1)

        if current_paragraph:  # Append the last paragraph if not empty
            paragraphs.append(current_paragraph.strip())

        return paragraphs

    def extract_english_second(self):
        print('--> Generating paragraph...')
        result = []
        basic_paragraph_list = self.extract_english_basic()
        total_paragraphs = len(basic_paragraph_list)
        with tqdm(total=total_paragraphs, desc='Processing paragraphs') as pbar:
            for paragraph in basic_paragraph_list[0:100]:
                proceed_paragraph = self.chinese_segment_process(paragraph)
                result.extend(proceed_paragraph)
                pbar.update(1)
        return result
    
    # Calculate cosine similarity between two sentences (query,document)
    def similarity_score(self, query, document):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        document_embedding = self.model.encode(document, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(query_embedding, document_embedding)
        return cosine_similarity

    ################################################################

    def generate_paragraph(self):
        # self.extract_english_basic()
        paragraphs = self.extract_paragraphs_from_pdf(self.en_filepath_in)
        save_list_to_file(paragraphs, './data/output/text.txt')
        extracted = len(paragraphs)
        print(extracted)
            