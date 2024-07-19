#!/bin/bash

# Echo the introductory text
echo "This script will run the setup and preprocessing scripts for our group."
echo "Group Members: Burak Yigit Uslu, Aleyna Sutbas, Ece Yilmaz"
echo "Now running the setup..."

# Run the setup Python script
python3 user_setup.py

# Notify that the setup script has completed
echo -e "\nSetup script complete!"
echo "The preprocessing script will now run with custom input articles."
echo "Our articles input data has 10 articles, 4 in English, 3 in German and 3 in Turkish. All articles are generated using generative AI by us."
echo "Now running the preprocessing script..."

# Run the preprocessing Python script with all input and output arguments
python3 user_preprocess.py \
    --input "./input_articles/article_de_1.json" \
    --input "./input_articles/article_de_2.json" \
    --input "./input_articles/article_de_3.json" \
    --input "./input_articles/article_en_1.json" \
    --input "./input_articles/article_en_2.json" \
    --input "./input_articles/article_en_3.json" \
    --input "./input_articles/article_en_4.json" \
    --input "./input_articles/article_tr_1.json" \
    --input "./input_articles/article_tr_2.json" \
    --input "./input_articles/article_tr_3.json" \
    --output "./preprocessed-articles"

# Notify that the preprocessing script has completed
echo -e "\nPreprocessing script complete!"
echo "You can now run your own queries using the user_inference.py, or run the RUN_EXAMPLE_QUERIES bash script."

# Additional instructions or next steps can be added here if necessary.
