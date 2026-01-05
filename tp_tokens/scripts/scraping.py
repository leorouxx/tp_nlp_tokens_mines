import argparse
import os

from ..scraping_data import extract_sentences

TXT_FILES = ["txt", "md"]
OUTPUT_FOLDER = 'output'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outfile")
    parser.add_argument("--datafile", default="")
    parser.add_argument("--datafolder", default="")
    parser.add_argument("--nbfiles", default=-1, type=int)
    args = parser.parse_args()

    data_files = []
    data_file = args.datafile
    out_file = args.outfile
    nb_files = int(args.nbfiles) 

    if data_file:
        data_files.append(data_file)
    
    else:
        data_folder : str = args.datafolder

        data_files = [
            os.path.join(data_folder, file) 
            for file in os.listdir(data_folder) 
            if file.split(".")[-1].lower() in TXT_FILES
            ]
        
        if nb_files > 0:
            data_files = data_files[:nb_files]


    if not data_files:
        print("No data provided, end of program")
        return 0
    
    import time
    start = time.time()

    nb_sentences = 0
    with open(out_file, 'w', encoding='utf-8') as f:
        for filepath in data_files:
            sentences = extract_sentences(filepath)
            nb_sentences += len(sentences)
            f.writelines(l + '\n' for l in sentences)
    
    end = time.time()

    print(f"{nb_sentences} sentences extracted.")
    print(f"In {(end - start):.2f} s")


