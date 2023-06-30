import jieba
import re
import torch
import json
import time
from common import save_list_to_file 
from langdetect import detect
from tqdm import tqdm
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, util

# Chinese - English Extracter

class Extractor :
    def __init__(self,en_filepath_in, cn_filepath_in,en_filepath_out ,cn_filepath_out,result_path_out,limit_score):
        self.cn_filepath_in = cn_filepath_in
        self.en_filepath_in = en_filepath_in
        self.cn_filepath_out = cn_filepath_out
        self.en_filepath_out = en_filepath_out
        self.result_path_out = result_path_out

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
        self.limit_score = limit_score   # Limit score of similarity 
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
        print('Loading Document...')
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
    
    # Calculate cosine similarity between two sentences (query,document)  NOT EFFICIENT SINGLE EMBEDDING
    def similarity_score(self, query, document):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        document_embedding = self.model.encode(document, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(query_embedding, document_embedding)
        return cosine_similarity
    
    # Pairing Chinese sentence list and English sentence list
    def similarity_pairing(self,chinese_list,english_list):
        pairing_list = []
        print("Chinese model encoding ...\n")
        corpus_embedding = self.model.encode(chinese_list, convert_to_tensor=True, batch_size= 64, show_progress_bar =True)
        results = []
        with tqdm(total=len(english_list), desc='Paring paragraphs') as pbar:
            for query in english_list:
                # Encoding query model (English)
                query_embedding = self.model.encode(query, convert_to_tensor=True)
                # Calculate cosine similarity between english and chinese embedding
                cos_scores = util.cos_sim(query_embedding, corpus_embedding)[0]

                # Select top score result from embedding model
                top_score, top_idx = torch.topk(cos_scores, k=1)
                
                pbar.update(1)
                # If score is less than LIMIT_SCORE, ignore the training data
                if round(top_score.item(), 3) < float(self.limit_score):
                    continue

                # JSON Formating of dictionary data
                result = {
                    "en": query,  # english sentence
                    "cn": chinese_list[top_idx.item()],
                    "score": round(top_score.item(), 3)
                }
                pairing_list.append(result)
                
        print("Successfully done ...\n")
        return pairing_list
    
    ################################################################

    def generate_paragraph(self):
        start_time = time.time()
        print("Generating chinese paragraphs...")
        chinese_paragraphs = self.extract_paragraphs_from_pdf(self.en_filepath_in)
        print("Generating English paragraphs...")
        english_paragraphs = self.extract_paragraphs_from_pdf(self.cn_filepath_in)
        print("Pairing...")
        pairing_result = self.similarity_pairing(chinese_paragraphs,english_paragraphs)

        with open(self.result_path_out, 'w', encoding="utf-8") as file:
            json.dump(pairing_result, file, ensure_ascii=False, indent=4)
        response_time = time.time() - start_time
        print("------------------------------------")
        print("Total Paragraphs: " + str(len(chinese_paragraphs)))
        print("Pairing Paragraphs: " +str(len(pairing_result)))
        print("Accuracy: " + str(float(self.limit_score) * 100)+"%")
        print("Percentages: "+str(len(pairing_result)/len(chinese_paragraphs)*100))
        print(f"Pairing time: {response_time} seconds")
        print("------------------------------------")
            