# This file will be called once the setup is complete. 
# It should perform all the preprocessing steps that are necessary to run the project. 
# All important arguments will be passed via the command line.
# The input files will adhere to the format specified in datastructure/input-file.json


# python user_preprocess.py --input "./sample_data/article_1.json" --input "./sample_data/article_2.json" --input "./sample_data/article_3.json" --input "./sample_data/article_4.json" --output "./preprocessed_articles/preprocessed_articles.json"

import json
from os.path import join, split as split_path
from langdetect import detect, DetectorFactory
from transformers import MarianMTModel, MarianTokenizer

import os
from os import listdir
from os.path import isfile, join, exists
import argparse
from os.path import exists

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
    with open(file_path, 'r') as f:
        data = json.load(f)

    article = "".join(data["content"])
    language = detect_language(article)

    if language == "de":
        translated_text = de_to_english(article)
    elif language == "tr":
        translated_text = tr_to_english(article)
    else:
        translated_text = article

    transformed_texts.append(translated_text)


    

# This is a useful argparse-setup, you probably want to use in your project:
parser = argparse.ArgumentParser(description='Preprocess the data.')
parser.add_argument('--input', type=str, help='Path to the input data.', required=True, action="append")
parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    output_file = args.output

    # Remove existing output file if it exists
    if exists(output_file):
        os.remove(output_file)

    transformed_texts = []

    for file_path in args.input:
        if isfile(file_path):
            handle_input_file(file_path, transformed_texts)
        else:
            print(f"File does not exist: {file_path}")

    # Create the output structure
    output_data = {
        "transformed_representation": transformed_texts
    }

    # Write the data to the output file
    # Write the data to the output file with improved readability
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4, separators=(',', ': '))

    print(f"Output has been written to {output_file}")
 