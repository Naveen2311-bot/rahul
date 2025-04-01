import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import time as t
import textwrap
import numpy as np
import csv
import os


st.markdown("""
    <h1 style="text-align: center; font-size: 65px; color: #4CAF50;">
    E-mail Spam Detection 
    </h1>
    """,
    unsafe_allow_html=True)

st.text("Let's go")
st.image("Image.jpg",width=500)

data = pd.read_csv("spam.csv")

data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham','Spam'],['Not Spam','Spam'])

mess = data['Message']
cat = data['Category']

(mess_train,mess_test,cat_train,cat_test) = train_test_split(mess, cat, test_size=0.2)

cv =CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

#creating Model

model = MultinomialNB()
model.fit(features, cat_train)

#Test Our model
features_test = cv.transform(mess_test)
#print(model.score(features_test,cat_test))
#predict Data
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result
    
st.markdown(
    """
    <div style="
        border: 1.3px solid #000080;  /* Single, solid black line */
        padding: 10px;           /* Space inside the border */
        margin: 10px;            /* Space outside the border */
    ">
    </div>
    """,
    unsafe_allow_html=True
)

input_mess = st.text_input('Enter Message Here')

if st.button('Validate'):
    output = predict(input_mess)
    st.markdown(output)
   
# Left align Image with Border
st.sidebar.markdown('<div class="center">',unsafe_allow_html=True)
st.sidebar.image("R1.png", width=170,)
st.sidebar.markdown('</div>',unsafe_allow_html=True)

st.sidebar.markdown("""
    <h1 style="text-align: left; font-family: 'Century'; border-bottom-style: solid; font-size: 23px; color: #ff00ff;">
       Developer : <strong>Naveen Kumar Thawait<strong>
    </h1>
    """,
    unsafe_allow_html=True)


st.sidebar.link_button("Connect Us", "https://www.google.com/search?q=naveen+kumar+thawait&oq=navee&gs_lcrp=EgZjaHJvbWUqDAgBECMYJxiABBiKBTIGCAAQRRg5MgwIARAjGCcYgAQYigUyDQgCEAAYkQIYgAQYigUyCggDEC4YsQMYgAQyBwgEEAAYgAQyCggFEC4YsQMYgAQyDQgGEC4YrwEYxwEYgAQyBggHEEUYQdIBCDI4MjRqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8")

# Streamlit UI
st.sidebar.markdown("""
    <h1 style="text-align: left; font-family: 'Elephant'; font-size: 45px; color: #FFD700;">
        Welcome to the Project... 
    </h1>
    """,unsafe_allow_html=True)

# Example of a rectangular border with CSS
st.sidebar.markdown(
    """
    <div style="
        border: 7px solid #00ff00; 
        border-radius: 5px;
        padding: 20px; 
        margin: 20px;
        background-color: #f9f9f9;
    ">
        <h3 style="color: #FF5733;">ABOUT OUR PROJECT</h3>
        <p style="color: #333;">
             "Unveiling the Power of AI: This intelligent Email Spam Detection system leverages advanced machine learning models to safeguard digital communication, ensuring a seamless and secure inbox experience—an innovation beyond conventional spam filters, redefining cybersecurity in the modern era."
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        border: 2px solid #000080;  /* Single, solid black line */
        padding: 10px;           /* Space inside the border */
        margin: 10px;            /* Space outside the border */
    ">
    </div>
    """,
    unsafe_allow_html=True
)

# Define CSV file name
file_name = "review_comments.csv"

# Function to save comments
def save_review(name, comment):
    # Create a DataFrame for new entry
    new_entry = pd.DataFrame([[name, comment]], columns=["Name", "Review Comment"])
    
    # Check if file exists
    if os.path.exists(file_name):
        existing_data = pd.read_csv(file_name)  # Load existing data
        updated_data = pd.concat([existing_data, new_entry], ignore_index=True)  # Append new data
    else:
        updated_data = new_entry  # Create new file with first entry

    # Save to CSV
    updated_data.to_csv(file_name, index=False)

# Streamlit App UI
st.markdown("""
    <h1 style="text-align: left; font-family: 'Cambria'; font-size: 27px; color: #000000;">
        Give your feedback
    </h1>
    """,
    unsafe_allow_html=True)

# Input Fields
name = st.text_input("Your Name")
comment = st.text_area("Enter Your Review Comment Here", height=90)

# Submit Button
if st.button("Submit Review"):
    if name and comment:
        save_review(name, comment)
        st.success("✅ Your review has been saved successfully!")
    else:
        st.warning("⚠️ Please fill all fields before submitting.")


st.toast("Welcome To Python Project!")



