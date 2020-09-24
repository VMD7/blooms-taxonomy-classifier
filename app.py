import streamlit as st
import joblib
import pickle
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.tokenize import punkt
nltk.download('wordnet')
from nltk.corpus.reader import wordnet
nltk.download('WordNetLemmatizer')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import matplotlib as plt


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
                           'BT5 - Evaluation': 4,
                           'BT6 - Creation': 5
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
            path_lr = 'Models/best_lrc.pickle'
            with open(path_lr, 'rb') as data:
                lr_model = pickle.load(data)

            path_mnb = 'Models/best_mnbc.pickle'
            with open(path_mnb, 'rb') as data:
                mnb_model = pickle.load(data)

            path_gbc = 'Models/best_gbc.pickle'
            with open(path_gbc, 'rb') as data:
                gbc_model = pickle.load(data)

            path_rfc = 'Models/best_rfc.pickle'
            with open(path_rfc, 'rb') as data:
                rfc_model = pickle.load(data)

            path_knn = 'Models/best_knnc.pickle'
            with open(path_knn, 'rb') as data:
                knn_model = pickle.load(data)

            path_svm = 'Models/best_svc.pickle'
            with open(path_svm, 'rb') as data:
                svc_model = pickle.load(data)
    
                   # Predict using the input model
            prediction_lr = lr_model.predict(create_features_from_text(text))[0]
            prediction_lr_proba = lr_model.predict_proba(create_features_from_text(text))[0]
            prediction_mnb = mnb_model.predict(create_features_from_text(text))[0]
            prediction_mnb_proba = mnb_model.predict_proba(create_features_from_text(text))[0]
            prediction_gbc = gbc_model.predict(create_features_from_text(text))[0]
            prediction_gbc_proba = gbc_model.predict_proba(create_features_from_text(text))[0]
            prediction_rfc = rfc_model.predict(create_features_from_text(text))[0]
            prediction_rfc_proba = svc_model.predict_proba(create_features_from_text(text))[0]
            prediction_knn = knn_model.predict(create_features_from_text(text))[0]
            prediction_knn_proba = svc_model.predict_proba(create_features_from_text(text))[0]
            prediction_svc = svc_model.predict(create_features_from_text(text))[0]
            prediction_svc_proba = svc_model.predict_proba(create_features_from_text(text))[0]
    
                    # Return result
            category_lr = get_category_name(prediction_lr)
            category_mnb = get_category_name(prediction_mnb)
            category_gbc = get_category_name(prediction_gbc)
            category_rfc = get_category_name(prediction_rfc)
            category_knn = get_category_name(prediction_knn)
            category_svc = get_category_name(prediction_svc)
            a=prediction_lr_proba.max()*100
            b=prediction_mnb_proba.max()*100
            c=prediction_gbc_proba.max()*100
            d=prediction_rfc_proba.max()*100
            e=prediction_knn_proba.max()*100
            f=prediction_svc_proba.max()*100
            best_one = {"category_lr":prediction_lr_proba.max()*100,"category_mnb":prediction_mnb_proba.max()*100,"category_gbc":prediction_gbc_proba.max()*100,"category_rfc":prediction_rfc_proba.max()*100,"category_knn":prediction_knn_proba.max()*100,"category_svc":prediction_svc_proba.max()*100}
            keymax = max(best_one, key = best_one.get)
            if keymax == "category_lr":
                return category_lr, best_one["category_lr"],a,b,c,d,e,f
            elif keymax == "category_mnb":
                return category_mnb,best_one["category_mnb"],a,b,c,d,e,f
            elif keymax == "category_gbc":
                return category_gbc,best_one["category_gbc"],a,b,c,d,e,f
            elif keymax == "category_rfc":
                return category_rfc,best_one["category_rfc"],a,b,c,d,e,f
            elif keymax == "category_knn":
                return category_knn,best_one["category_knn"],a,b,c,d,e,f
            else:
                return category_svc,best_one["category_svc"],a,b,c,d,e,f

    
        
        st.info("Prediction with Various Models")
        
        bt_text = st.text_area("Question to Predict","Typer Here")

        if st.button("Classify"):
            st.text("Original Text ::\n{}".format(bt_text))
          


            prediction = predict_from_text(bt_text)

            st.success("Blooms Taxonomy Level   ::   {}".format(prediction[0]))
            st.success("Maximum Probability   ::   {}".format(prediction[1]))
            st.write("Performance of Various Algorithms")

            data = pd.DataFrame({
                'Various Algorithm': ['Logistic Regression', 'Multinomial Naive Bayes', 'Gradient Boosting Classifier','Random Forest Classifier','k-Nearest Neighbors','Support Vector Machine'],
                'Maximum Accuracy': [(prediction[2]),prediction[3],prediction[4],prediction[5],prediction[6],prediction[7]],
            }).set_index('Various Algorithm')

            st.write(data)
            st.bar_chart(data)



    if choice == "About":
        st.success("This web app is developed by Vijay Devane.")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

if __name__ =='__main__':
    main()
