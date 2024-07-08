# This file is used to setup the project. It is executed when the project is imported.
# This file should be used to download all large files (e.g., model weights) and store them to disk.
# In this file, you can also check if the environment works as expected.
# If something goes wrong, you can exit the script with a non-zero exit code.
# This will help you detect issues early on.
#
# Below, you can find some sample code:

# our code
import os
import sys
import transformers
from transformers import BertTokenizer, BertModel, GPTNeoModel, GPT2Tokenizer, LlamaModel, LlamaTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import sentencepiece
import json
import langdetect

def download_model(model_name, model_class, tokenizer_class):
    """
    Downloads a model and its tokenizer if not already cached.
    """
    try:
        model = model_class.from_pretrained(model_name)
        tokenizer = tokenizer_class.from_pretrained(model_name)
        print(f"Downloaded and cached {model_name}")
        return True
    except Exception as e:
        print(f"Failed to download {model_name}: {e}")
        return False


def download_large_files():
    """
    Downloads all required large files such as model weights.
    """

    relative_local_bert_path = 'fine-tuned-BERT'
    # Download our fine-tuned BERT
    if not download_model(relative_local_bert_path, BertModel, BertTokenizer):
        return False
    
    relative_local_llama3_path = 'fine-tuned-llama3'
    if not download_model(relative_local_llama3_path, LlamaModel, LlamaTokenizer):
        return False

    # scrapped from the final pipeline
    # Downloading GPT-Neo 2.7B
    # if not download_model('EleutherAI/gpt-neo-2.7B', GPTNeoModel, GPT2Tokenizer):
    #     return False
    
    # Download translation models for each required language
    for lang in ['de', 'tr']:
        model_name = f'Helsinki-NLP/opus-mt-{lang}-en'
        if not download_model(model_name, AutoModelForSeq2SeqLM, AutoTokenizer):
            return False

    return True

def check_environment():
    try:
        import transformers
        import torch
        import sentencepiece
        print("All required libraries are installed.")
        return True
    except ImportError as e:
        print(f"Required library not installed: {e}")
        return False


if __name__ == "__main__":
    print("user_setup.py is running.")
    
    if not check_environment():
        print("Environment check failed.")
        exit(1)
        
    if not download_large_files():
        print("Downloading large files failed.")
        exit(1)
    
    print("user_setup.py is completed.")
    exit(0)