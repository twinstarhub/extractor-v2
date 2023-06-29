import os
from Extractor import Extractor 
from dotenv import load_dotenv
load_dotenv('.env')

en_file_path_in = os.environ['INPUT_EN_FILE_PATH']
cn_file_path_in = os.environ['INPUT_CN_FILE_PATH']
en_file_path_out = os.environ['OUTPUT_EN_FILE_PATH']
cn_file_path_out = os.environ['OUTPUT_CN_FILE_PATH']
test_path = os.environ['OUTPUT_TEST_FILE_PATH']


# This is a English to Chinese matching script
if __name__ == '__main__':

    tw_extractor = Extractor(en_file_path_in,cn_file_path_in,en_file_path_out,cn_file_path_out)
    tw_extractor.generate_paragraph()
    print('This is matching script')

