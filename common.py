def save_list_to_file(list,output_file_path):
    # Provide the path to the output text file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for sentence in list:
            file.write(sentence + '\n\n')