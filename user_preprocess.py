# This file will be called once the setup is complete. 
# It should perform all the preprocessing steps that are necessary to run the project. 
# All important arguments will be passed via the command line.
# The input files will adhere to the format specified in datastructure/input-file.json

import json
from os.path import join, split as split_path
from langdetect import detect, DetectorFactory
from transformers import MarianMTModel, MarianTokenizer

# Setup the language detection
DetectorFactory.seed = 0

def de_to_english(text):
    de_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    de_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    inputs = de_en_tokenizer(text, return_tensors="pt", padding=True)
    outputs = de_en_model.generate(**inputs)
    return de_en_tokenizer.decode(outputs[0], skip_special_tokens=True)

def tr_to_english(text):
    tr_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tr-en")
    tr_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tr-en")
    inputs = tr_en_tokenizer(text, return_tensors="pt", padding=True)
    outputs = tr_en_model.generate(**inputs)
    return tr_en_tokenizer.decode(outputs[0], skip_special_tokens=True)

# This function should return the detected language of the input text
def detect_language(text):
    accepted_languages = ["de", "tr", "en"]
    if detect(text) in accepted_languages:
        return detect(text)
    else:
        return "en"

# TODO Implement the preprocessing steps here
def handle_input_file(file_location, output_path):
    with open(file_location, 'r') as f:
        data = json.load(f)
    
    # Detect the language of the articles and translate them to English
    article = "".join(data["content"])
    language = detect_language(article)

    if language == "de":
        translated_text = de_to_english(article)
    elif language == "tr":
        translated_text = tr_to_english(article)
    else:
        translated_text = article

    transformed_data = {
        # "title": data["title"],
        # "timestamp": data["timestamp"],
        "transformed_representation": [translated_text]
    }

    file_name = split_path(file_location)[-1]
    output_file_path = join(output_path, file_name)
    with open(output_file_path, 'w') as f:
        json.dump(transformed_data, f)

    

# This is a useful argparse-setup, you probably want to use in your project:
import argparse
parser = argparse.ArgumentParser(description='Preprocess the data.')
parser.add_argument('--input', type=str, help='Path to the input data.', required=True, action="append")
parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    files_inp = args.input
    files_out = args.output
    
    for file_location in files_inp:
        handle_input_file(file_location, files_out)

 