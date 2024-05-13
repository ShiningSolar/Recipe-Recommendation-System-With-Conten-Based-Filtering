import pickle
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.header("Recipe Recommendation System")
dataset = pickle.load(open('dataset.pkl', 'rb'))

# Function to calculate cosine similarity between two vectors
def calculate_cosine_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return None
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Function to generate recommendations based on cosine similarity
def generate_recommendations(query_embedding, df, top_n=5):
    similarities = df['embedding'].apply(lambda x: calculate_cosine_similarity(query_embedding, x))
    df['similarity'] = similarities
    recommendations = df.sort_values(by='similarity', ascending=False).head(top_n)
    return recommendations.drop(columns=['embedding', 'similarity', 'RecipeId'])

# Example usage: generate recommendations based on the first row embedding
query_embedding = dataset['embedding'][1]
recommendations = generate_recommendations(query_embedding, dataset)
#print("Recommendations based on the first row embedding:")

df = recommendations
# Display the DataFrame without index
df = df.set_index('*',drop=True)
st.dataframe(df)
