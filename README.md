# Recipe-Recommendation-System-With-Content-Based-Filtering
This repo is recipe recommendation system based on recipe's ingredients, recipe category, and keywords.
Content-based Filtering used for the recommendation system
Data collection used in this repo is recipe.csv from ( https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews  ) with several data cleaning including removing missing value from item and dropping unused colums
Dataframe that have been cleaned will be exported to a pickle file
Word2Vec used for word embedding recipe's ingredients, recipe category, and keywords for recommendation.
cosine similarity used for measuring similarity between texts that have been embedded
this repo will be deploy on streamlit.



