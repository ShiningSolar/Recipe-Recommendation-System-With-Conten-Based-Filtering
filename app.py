import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

st.header("Recipe Recommendation System")
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
dataset = pickle.load(open('dataset.pkl', 'rb'))
image = pickle.load(open('image.pkl', 'rb'))
ingredient = pickle.load(open('ingredient.pkl', 'rb'))
info = pickle.load(open('info.pkl', 'rb'))
recipeName = pickle.load(open('recipeName.pkl', 'rb'))

#feecth image
def fecth_image(similar_items):
    recipe_name = []
    recipe_index = []
    recipe_image = []
    #for index in similar_items:


# Function to generate recommendations based on knn
def generate_knn_recommendations(name, df, knn_model, n_neighbors=10):
    #item_index = df[df['item_id'] == item_id].index[0]
    item_index = df[df['Name'] == name].index[0]
    distances, indices = knn_model.kneighbors(tfidf[item_index], n_neighbors=n_neighbors + 1)
    similar_items = df.iloc[indices[0][1:]]  # Menghapus item itu sendiri dari hasil
    return similar_items

@st.experimental_fragment
def fragment_function():
    selected_recipe = st.selectbox(
        "Type or select a recipe",
        recipeName,
        None,
        placeholder = "Type or select recipe",
        label_visibility = "collapsed"
    )
    if st.button('Show Recommendation'):
        #st.write(selected_recipe)
        recommendations = generate_knn_recommendations(selected_recipe, info, model)
        st.dataframe(recommendation)
        #column = st.columns(10)
        
fragment_function()


    
    

# Contoh penggunaan: merekomendasikan item berdasarkan item_id 1
#recommendations = generate_knn_recommendations(96, info, model)
#print("Recommendations based on item_id 1:")

#df = recommendations
# Display the DataFrame without index
#df = df.style.hide_index()



