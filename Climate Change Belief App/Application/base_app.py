"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from streamlit_option_menu import option_menu

# Natural Langauge Processing

import nltk
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import string
import re

import spacy
nlp=spacy.load('en_core_web_sm')

# Data dependencies
import pandas as pd

# Tokenization & Lematization

stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
whitelist=["n't", "not"]

def tokenize(sentence):
	tokenized_sentence = word_tokenize(sentence)
	return tokenized_sentence
            
def remove_stopwords(sentence):
	filtered_sentence = []
	for w in sentence:
		if w not in stopwords and len(w) > 1 and w[:2] != '//' and w != 'https' or w in whitelist: 
			filtered_sentence.append(w)
	return filtered_sentence
    
def lemmatize(sentence):
	return [lemmatizer.lemmatize(word) for word in sentence]
    
def join_to_string(sentence): 
	return ' '.join(sentence)


# Vectorizer
news_vectorizer = open("resources/vectorizer.pkl","rb")# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer) 

#Load Logo image
logo=open("resources/LOGO.jpeg")

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )


# Load your raw data
raw = pd.read_csv("resources/train.csv")

def get_keys(val, my_dict):
	for key, value in my_dict.items():
		if val==value:
			return key
# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	
	with st.sidebar:
		selection=option_menu(
			menu_title="Main Menu",
			options=["Prediction", "Information","NLP","Contact Us","About Us"],
			icons=["bar-chart-line","info-circle","arrow-repeat","envelope","people-fill"],
			menu_icon="cast",
			default_index=0,
		)
	st.sidebar.image("resources/LOGO.jpeg",use_column_width=True)
	# Building out the "Information" page
	if selection == "Information":
		st.title("Tweet Classifer")
		st.subheader("Climate change tweet classification")
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		#st.info("Prediction with ML Models")
		# Creates a main title and subheader on your page -
		# these are static across all pages
		st.title("Tweet Classifer")
		st.subheader("Climate change tweet classification")
		
		select_model=option_menu(
			menu_title="Select Classifier",
			options=["Decision Tree", "KNeighbors","Logistic_Regression","SVC"],
			icons=["arrow-right-circle","arrow-right-circle","arrow-right-circle","arrow-right-circle"],
			menu_icon="hand-index-thumb-fill",
			default_index=0,		
		)

		
		if select_model == "Decision Tree":
			predictor = joblib.load(open(os.path.join("resources/DecisionTreeClassifier.pkl"),"rb"))

		if select_model == "KNeighbors":
			predictor = joblib.load(open(os.path.join("resources/KNeighborsClassifier.pkl"),"rb"))

		if select_model =="Logistic_Regression":
			predictor = joblib.load(open(os.path.join("resources/Logistic_Regression.pkl"),"rb"))

		if select_model == "SVC":
			predictor = joblib.load(open(os.path.join("resources/SVC.pkl"),"rb"))

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","")
		prediction_lables={'Negative':-1, "Positive": 1, "Neutral":0, "News":2}
		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()	
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			final_result=get_keys(prediction, prediction_lables)
			st.success("Text Categorized as:: {}".format(final_result))
			st.markdown("**_Click NLP if you want to see the Tokenizaion and Lemmatization process_**")
	if selection == "NLP":
		st.title("Natural Language Processing")
		tweets_text=st.text_area("Tweet here", "")
		nlp_task=["Tokenization", "Lemmatization"]
		task_choice=st.selectbox("Choose NLP Task", nlp_task)
		if st.button("Analyze"):
			st.info("Original text {}".format(tweets_text))

			docx=nlp(tweets_text)
			if task_choice=='Tokenization':
				result=[token.text for token in docx]
				st.json(result)

			elif task_choice=="Lemmatization":
				result=["'Token':{}, 'Lemma':{}".format(token.text, token.lemma_) for token in docx ]
				st.json(result)

	if selection == "Contact Us":
		st.title(":mailbox: Get In Touch With Us!")
		contact_form = """
				<form action="https://formsubmit.co/mikelacoste25@gmail.com" method="POST">
					<input type="hidden" name="_captcha" value="false">
					<input type="text" name="name" placeholder="Your name" required>
					<input type="email" name="email" placeholder="Your email" required>
					<textarea name="message" placeholder="Your message here"></textarea>
					<button type="submit">Send</button>
				</form>
				"""

		st.markdown(contact_form, unsafe_allow_html=True)

				# Use Local CSS File
		def local_css(file_name):
			with open(file_name) as f:
				st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

		local_css("style/style.css")
	if selection == "About Us":
		#add_bg_from_local('LOGO.jpeg')
		st.title("Who we are")
		st.header("@MLCorp")
		st.markdown("**_Innovations designed for creating a better world_**")
		st.write("We specialize in Machine & Deep Learning.")
		st.subheader("Machine Learning Techniques:")
		st.write("Advanced Regression, Classification, & Unsupervised learning")
		st.subheader("Deep Learning Techniques:")
		st.write("Neural Network") 

		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
