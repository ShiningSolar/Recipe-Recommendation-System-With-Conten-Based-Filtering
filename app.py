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
        url = url[0]
        #mengecek apakah url image lebih dari 1 item
        #if len(url) > 1:
        #    url = url[0]
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

@st.experimental_dialog("Recipe details")
def recipe_details(item):
    st.write(item)

@st.experimental_fragment
def fragment_function():
    selected_recipe = st.selectbox(
        "Type or select a recipe",
        recipeName,
        None,
        placeholder = "Type or select recipe",
        label_visibility = "collapsed"
    )
    button = st.empty()
    recommendation_box = st.empty()
    if button.button('Show Recommendation'):
        #st.write(selected_recipe)
        recommendations, recipe_image, recipe_name = generate_knn_recommendations(selected_recipe, info, model)
        #st.dataframe(recommendations)
        #recommendation_box = st.empty()
        #column = st.columns(10)
        with recommendation_box.container():
            row1 = st.columns(2, gap = "medium")
            row2 = st.columns(2, gap = "medium")
            row3 = st.columns(2, gap = "medium")
            index = 0
            for tile in row1 + row2 + row3:
                tile = tile.columns(2)
                tile[0] = tile[0].image(recipe_image[index])
                #tile[1] = tile[1].link_button(recipe_name[index], "https://recipe-recommendation-system-with-content-based-filtering-1008.streamlit.app/recipe_page")
                #tile[1] = tile[1].page_link("pages/recipe_page.py", label=recipe_name[index], use_container_width = True)
                #page_button = tile[1].empty()
                name = recipe_name[index]
                if "recipe_details" not in st.session_state:
                    if tile[1].button(label = name):
                        recipe_details(name)
                index = index + 1
    if selected_recipe == '':
        recommendation_box.empty()
        
fragment_function()


    
    

# Contoh penggunaan: merekomendasikan item berdasarkan item_id 1
#recommendations = generate_knn_recommendations(96, info, model)
#print("Recommendations based on item_id 1:")

#df = recommendations
# Display the DataFrame without index
#df = df.style.hide_index()



