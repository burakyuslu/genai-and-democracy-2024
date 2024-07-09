# This file will be executed when a user wants to query your project.
import argparse
from os.path import join
import os
import json
from langdetect import detect, DetectorFactory
from transformers import MarianMTModel, MarianTokenizer, BertModel, BertTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load BERT model and tokenizer
tokenizer_bert = BertTokenizer.from_pretrained('fine-tuned-BERT')
model_bert = BertModel.from_pretrained('fine-tuned-BERT')
# load LLAMA model and tokenizer
tokenizer_llama = AutoTokenizer.from_pretrained('fine-tuned-llama3')
model_llama = AutoModelForSeq2SeqLM.from_pretrained('fine-tuned-llama3')

# Setup the language detection
DetectorFactory.seed = 0

# The number of articles to be returned
ARTICLE_COUNT_TO_RETURN = 3

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

def detect_language(text):
    accepted_languages = ["de", "tr", "en"]
    if detect(text) in accepted_languages:
        return detect(text)
    else:
        return "en"

# Define function for text generation
def get_query_suggestions(original_query, max_length=50, num_return_sequences=5):
    inputs = tokenizer_llama(original_query, return_tensors="pt")

    # Generate alternative queries
    generated_ids = model_llama.generate(inputs.input_ids.to(model_llama.device),
                                   max_length=max_length,
                                   num_return_sequences=num_return_sequences,
                                   num_beams=num_return_sequences)  # Set num_beams equal to num_return_sequences

    alternative_queries = [tokenizer_llama.decode(g, skip_special_tokens=True) for g in generated_ids]
    return alternative_queries

# TODO Implement the inference logic here
def handle_user_query(query, query_id, output_path):
    result = dict()

    # Detect the language of the query, translate it
    result["detected_language"] = detect_language(query)
    translated_query = ""
    if result["detected_language"] == "de":
        translated_query = de_to_english(query)
    elif result["detected_language"] == "tr":
        translated_query = tr_to_english(query)
    else:
        translated_query = query

    # Load the preprocessed articles
    preprocessed_articles_path = "preprocessed_articles/preprocessed_articles.json"
    article_representations = []
    with open(preprocessed_articles_path, "r") as f:
        article_representations = json.load(f)["transformed_representation"]
    
    # Rank the articles
    ranked_articles = rank_articles(translated_query, article_representations)
    result["generated_queries"] = ranked_articles

    # Get query suggestions
    query_suggestions = get_query_suggestions(original_query=query)
    result["query_suggestions"] = query_suggestions
    
    # Save the result
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(join(output_path, f"{query_id}.json"), "w") as f:
        json.dump(result, f)


def encode_text(text):
    inputs = tokenizer_bert(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    outputs = model_bert(**inputs)
    # Get the embeddings for the first token (CLS token)
    return outputs.last_hidden_state[:,0,:].detach().numpy().reshape(1, -1)

# TODO OPTIONAL
# This function is optional for you
# You can use it to interfer with the default ranking of your system.
#
# If you do embeddings, this function will simply compute the cosine-similarity
# and return the ordering and scores
def rank_articles(generated_queries, article_representations):
    """
    This function takes as arguments the generated / augmented user query, as well as the
    transformed article representations.
    
    It needs to return a list of shape (M, 2), where M <= #article_representations.
    Each tuple contains [index, score], where index is the index in the article_repr array.
    The list need already be ordered by score. Higher is better, between 0 and 1.
    
    An empty return list indicates no matches.
    """
    # encode all articles and create  embeddings
    article_embeddings = np.array([encode_text(article) for article in article_representations])
    
    # encode the generated queries and create embeddings
    query_embedding = encode_text(generated_queries)

    # calculate cosine similarity between the query and all article embeddings
    similarities = cosine_similarity(query_embedding, article_embeddings.reshape(-1, article_embeddings.shape[-1]))
    ranked_articles = np.argsort(similarities[0])[::-1]

    # return the ranked articles
    return [article_representations[idx] for idx in ranked_articles]

# This is a sample argparse-setup, you probably want to use in your project:
parser = argparse.ArgumentParser(description='Run the inference.')
parser.add_argument('--query', type=str, help='The user query.', required=True, action="append")
parser.add_argument('--query_id', type=str, help='The IDs for the queries, in the same order as the queries.', required=True, action="append")
parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    queries = args.query
    query_ids = args.query_id
    output = args.output
    
    assert len(queries) == len(query_ids), "The number of queries and query IDs must be the same."
    
    for query, query_id in zip(queries, query_ids):
        handle_user_query(query, query_id, output)
    