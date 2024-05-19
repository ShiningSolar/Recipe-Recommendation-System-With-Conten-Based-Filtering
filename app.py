import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

st.header("Recipe Recommendation System")
model = pickle.load(open('model.pkl', 'rb'))
tfidf_matrix = pickle.load(open('tfidf.pkl', 'rb'))
dataset = pickle.load(open('dataset.pkl', 'rb'))
image = pickle.load(open('image.pkl', 'rb'))
ingredient = pickle.load(open('ingredient.pkl', 'rb'))
info = pickle.load(open('info.pkl', 'rb'))
recipeName = pickle.load(open('recipeName.pkl', 'rb'))

#fecth image
def fecth_image(df):
    #list untuk menyimpan url image setiap resep
    recipe_image = []
    recipe_name = []
    
    
    for recipe in df['Name']:
        #mendapatkan index berdasarkan nama resep
        index = df.loc[df['Name'] == recipe].index[0]
        recipe_name.append(str(recipe))
        url = image.Images[index]
        #mengecek apakah url image lebih dari 1 item
        if len(url) > 1:
            url = url[0]
        #menyimpan url pada list
        recipe_image.append(str(url))

    return recipe_image, recipe_name

# Function to generate recommendations based on knn
def generate_knn_recommendations(name, df, knn_model, n_neighbors=10):
    #item_index = df[df['item_id'] == item_id].index[0]
    item_index = df[df['Name'] == name].index[0]
    distances, indices = knn_model.kneighbors(tfidf[item_index], n_neighbors=n_neighbors + 1)
    similar_items = df.iloc[indices[0][1:]]  # Menghapus item itu sendiri dari hasil
    recipe_image, recipe_name = fecth_image(similar_items)
    return similar_items, recipe_image, recipe_name

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
        recommendations, recipe_image, recipe_name = generate_knn_recommendations(selected_recipe, info, model)
        #st.dataframe(recommendations)
        recommendation_box = st.empty()
        #column = st.columns(10)
        with recommendation_box.container(height = 450):
            row1, row2, row3 = st.columns(2)
            index = 0
            for tile in row1 + row2 + row3:
                tile = st.columns(2)
                tile[0] = st.image(recipe_image[index])
                tile[1] = st.markdown(recipe_name[index])
                index = index + 1
        
fragment_function()


    
    

# Contoh penggunaan: merekomendasikan item berdasarkan item_id 1
#recommendations = generate_knn_recommendations(96, info, model)
#print("Recommendations based on item_id 1:")

#df = recommendations
# Display the DataFrame without index
#df = df.style.hide_index()



