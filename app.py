import streamlit as st
import joblib
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import punkt
from nltk.corpus.reader import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy


def main():
    """ Blooms Taxonomy classifier"""
    st.title("Blooms Taxonomy Classifier")
    st.subheader("ML App for Blooms Taxonomy Level Prediction")
    
    activities = ["Prediction","About"]
    choice =st.sidebar.selectbox("Choose Activity",activities )
   
    if choice == "Prediction":
        path_tfidf = "Models/tfidf.pickle"
                
        with open(path_tfidf, 'rb') as data:
            tfidf = pickle.load(data)
        category_codes = {
                           'BT1 - Knowledge': 0,
                           'BT2 - Comprehension': 1,
                           'BT3 - Application': 2,
                           'BT4 - Analysis': 3,
                           'BT5 - Synthesis': 4,
                           'BT6 - Evaluation': 5
                           }

        punctuation_signs = list("?:!.,;")
        stop_words = list(stopwords.words('english'))

        def create_features_from_text(text):
    
                      # Dataframe creation
            lemmatized_text_list = []
            df = pd.DataFrame(columns=['Questions'])
            df.loc[0] = text
            df['Questions_Parsed_1'] = df['Questions'].str.replace("\r", " ")
            df['Questions_Parsed_1'] = df['Questions_Parsed_1'].str.replace("\n", " ")
            df['Questions_Parsed_1'] = df['Questions_Parsed_1'].str.replace("    ", " ")
            df['Questions_Parsed_1'] = df['Questions_Parsed_1'].str.replace('"', '')
            df['Questions_Parsed_2'] = df['Questions_Parsed_1'].str.lower()
            df['Questions_Parsed_3'] = df['Questions_Parsed_2']
            for punct_sign in punctuation_signs:
                df['Questions_Parsed_3'] = df['Questions_Parsed_3'].str.replace(punct_sign, '')
            df['Questions_Parsed_4'] = df['Questions_Parsed_3'].str.replace("'s", "")
            wordnet_lemmatizer = WordNetLemmatizer()
            lemmatized_list = []
            text = df.loc[0]['Questions_Parsed_4']
            text_words = text.split(" ")
            for word in text_words:
                lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
            lemmatized_text = " ".join(lemmatized_list)    
            lemmatized_text_list.append(lemmatized_text)
            df['Questions_Parsed_5'] = lemmatized_text_list
            df['Questions_Parsed_6'] = df['Questions_Parsed_5']
            for stop_word in stop_words:
                regex_stopword = r"\b" + stop_word + r"\b"
                df['Questions_Parsed_6'] = df['Questions_Parsed_6'].str.replace(regex_stopword, '')
            df = df['Questions_Parsed_6']
            df = df.rename({'Questions_Parsed_6': 'Questions_Parsed'})
    
            # TF-IDF
            features = tfidf.transform(df).toarray()
    
            return features
                
        def get_category_name(category_id):
            for category, id_ in category_codes.items():    
                if id_ == category_id:
                    return category                
        def predict_from_text(text):
    
                   # Predict using the input model
            prediction_svc = svc_model.predict(create_features_from_text(text))[0]
            prediction_svc_proba = svc_model.predict_proba(create_features_from_text(text))[0]
    
                    # Return result
            category_svc = get_category_name(prediction_svc)
    
            return category_svc , prediction_svc_proba.max()*100

    
        
        st.info("Prediction with Various Models")
        
        bt_text = st.text_area("Question to Predict","Typer Here")
        all_ml_models = ["Logistic Regression", "Multinomial Naive Bayes","Gradient Boosting","Random Forest","KNN","SVM"]
        model_choice = st.selectbox("Choose ML Model",all_ml_models)
        Prediction_labels = {"BT1":1,"BT2":2,"BT3":3,"BT4":4,"BT5":5,"BT6":6}
        if st.button("Classify"):
            st.text("Original Text ::\n{}".format(bt_text))
            if model_choice == "Logistic Regression":
                path_svm = 'Models/best_lrc.pickle'
                with open(path_svm, 'rb') as data:
                    svc_model = pickle.load(data)
                

            if model_choice == "Multinomial Naive Bayes":
                path_svm = 'Models/best_mnbc.pickle'
                with open(path_svm, 'rb') as data:
                    svc_model = pickle.load(data)

            if model_choice == "Gradient Boosting":
                path_svm = 'Models/best_gbc.pickle'
                with open(path_svm, 'rb') as data:
                    svc_model = pickle.load(data)

            if model_choice == "Random Forest":
                path_svm = 'Models/best_rfc.pickle'
                with open(path_svm, 'rb') as data:
                    svc_model = pickle.load(data)


            if model_choice == "KNN":
                path_svm = 'Models/best_knnc.pickle'
                with open(path_svm, 'rb') as data:
                    svc_model = pickle.load(data)

            if model_choice == "SVM":
                path_svm = 'Models/best_svc.pickle'
                with open(path_svm, 'rb') as data:
                    svc_model = pickle.load(data)

            prediction = predict_from_text(bt_text)

            st.success("Blooms Taxonomy Level   ::   {}".format(prediction[0]))
            #st.write("Probability = ",prediction[1])

    if choice == "About":
        st.success("This web app is developed by Vijay Devane.")
    
if __name__ =='__main__':
    main()
