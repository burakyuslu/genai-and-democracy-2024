#!/bin/bash

# Echo the introduction and important reminders
echo "This script runs 3 example user queries to demonstrate how the inference script is working."
echo "This script should ONLY be run AFTER the RUN_SETUP_PREPROCESS.sh script has been run."
echo "Group Members: Burak Yigit Uslu, Aleyna Sutbas, Ece Yilmaz"

# Running the first example query
echo -e "\nRunning the first example query."
echo "Search topic: authentic cuisine"
# Run the inference script for the first query
python3 user_inference.py --user_query "authentic cuisine" --output_file "query_results" --query_id "example_query_1"
echo "The first search query is run, the results are saved to query_results/example_query_1.json"

# Running the second example query
echo -e "\nRunning the second example query."
echo "Search topic: travelling"
# Run the inference script for the second query
python3 user_inference.py --user_query "travelling" --output_file "query_results" --query_id "example_query_2"
echo "The second search query is run, the results are saved to query_results/example_query_2.json"

# Running the third example query
echo -e "\nRunning the third example query."
echo "Search topic: computer science"
# Run the inference script for the third query
python3 user_inference.py --user_query "computer science" --output_file "query_results" --query_id "example_query_3"
echo "The third search query is run, the results are saved to query_results/example_query_3.json"

# Final statement
echo -e "\nAll example queries are run!"
echo "The results are saved to the query_results directory."
