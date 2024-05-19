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

st.session_state
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
        #menyimpan url pada list
        recipe_image.append(str(url))

    return recipe_image, recipe_name

# Function to generate recommendations based on knn
def generate_knn_recommendations(name, df, knn_model, n_neighbors=10):
    item_index = df[df['Name'] == name].index[0]
    distances, indices = knn_model.kneighbors(tfidf[item_index], n_neighbors=n_neighbors + 1)
    similar_items = df.iloc[indices[0][1:]]  # Menghapus item itu sendiri dari hasil
    recipe_image, recipe_name = fecth_image(similar_items)
    return similar_items, recipe_image, recipe_name

@st.experimental_dialog("Recipe Details",width = "large")
def recipe_details(name, image, index):
    st.image(image)
    st.write(name)

@st.experimental_fragment
def fragment_function():
    recommendations = st.session_state.recommendations
    recipe_image = st.session_state.recipe_image
    recipe_name = st.session_state.recipe_name
    texttest = st.empty()

    recommendation_box = st.empty()

    with recommendation_box.container():
        row1 = st.columns(2, gap = "medium")
        row2 = st.columns(2, gap = "medium")
        row3 = st.columns(2, gap = "medium")
        if 'index' not in st.session_state:
            st.session_state['index'] = 0
        index = st.session_state['index']
        
        for tile in row1 + row2 + row3:
            tile = tile.columns(2)
            key_image = 'image' + str(index)
            if key_image not in st.session_state:
                st.session_state[key_image] = recipe_image[index]
            tile[0] = tile[0].image(recipe_image[index])
            name = str(recipe_name[index])
            key_name = 'name'+str(index)
            if key_name not in st.session_state:
                result = tile[1].button(label = name, key = key_name)
            #if "recipe_details" not in st.session_state:
            tile[1].write(result)
            if result:
                texttest.write('success')
                recipe_details(st.session_state[key_name], st.session_state[key_image], index)
            st.session_state.index = index + 1
            
    if st.button('test'):
        recipe_details('test')
        
def searchbox_view():
    selected_recipe = st.selectbox(
        "Type or select a recipe",
        recipeName,
        None,
        placeholder = "Type or select recipe",
        label_visibility = "collapsed"
    )
    button = st.button('Show Recommendation')
    if button:
        recommendations, recipe_image, recipe_name = generate_knn_recommendations(selected_recipe, info, model)
        if 'recommendations' not in st.session_state:
            st.session_state['recommendations']=recommendations
        if 'recipe_image' not in st.session_state:
            st.session_state['recipe_image']=recipe_image
        if 'recipe_name' not in st.session_state:
            st.session_state['recipe_name']=recipe_name
        fragment_function()

searchbox_view()
#fragment_function()


    
    

# Contoh penggunaan: merekomendasikan item berdasarkan item_id 1
#recommendations = generate_knn_recommendations(96, info, model)
#print("Recommendations based on item_id 1:")

#df = recommendations
# Display the DataFrame without index
#df = df.style.hide_index()



