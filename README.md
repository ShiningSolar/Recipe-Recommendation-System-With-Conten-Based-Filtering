# Recipe-Recommendation-System-With-Content-Based-Filtering
This repo is recipe recommendation system based on recipe's ingredients ( maybe will be added recipe category, and keywords.)

Content-based Filtering used for the recommendation system

Data collection used in this repo is recipe.csv from ( https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews  ) with several data cleaning including removing missing values from items and dropping unused columns

Dataframe that have been cleaned will be exported to a pickle file

Word2Vec is used for word embedding recipe's ingredients, recipe category, and keywords for recommendation.

cosine similarity is used for measuring similarity between texts that have been embedded

this repo will be deployed on Streamlit. ( https://recipe-recommendation-system-with-content-based-filtering.streamlit.app/ )



