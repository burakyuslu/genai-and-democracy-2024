# This file will be called once the setup is complete. 
# It should perform all the preprocessing steps that are necessary to run the project. 
# All important arguments will be passed via the command line.
# The input files will adhere to the format specified in datastructure/input-file.json

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
    with open(file_location, 'r') as f:
        data = json.load(f)

    article = "".join(data["content"])
    language = detect_language(article)

    if language == "de":
        translated_text = de_to_english(article)
    elif language == "tr":
        translated_text = tr_to_english(article)
    else:
        translated_text = article

    transformed_data = {
        "transformed_representation": translated_text
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    mode = 'a' if exists(output_file) else 'w'
    with open(output_file, mode) as f:
        if mode == 'a':
            f.seek(0, os.SEEK_END)
            f.seek(f.tell() - 1, os.SEEK_SET)
            f.truncate()  # Remove the last closing bracket
            f.write(',')  # Prepare to append new JSON object
        else:
            f.write('[')  # Start an array if file is new
        json.dump(transformed_data, f)
        f.write(']')


    

# This is a useful argparse-setup, you probably want to use in your project:
parser = argparse.ArgumentParser(description='Preprocess the data.')
parser.add_argument('--input', type=str, help='Path to the input data.', required=True, action="append")
parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    output_file = args.output

    print(f"Input Files: {args.input}")
    print(f"Output File: {output_file}")
    print(f"Input 1: {args.input[0]}")
    print(f"Input 2: {args.input[1]}")


    for file_path in args.input:
        if isfile(file_path):
            handle_input_file(file_path, output_file)
        else:
            print(f"File does not exist: {file_path}")
 