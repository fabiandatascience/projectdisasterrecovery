import sys

import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages_classified', con=engine)
    X = df['message']
    X = X.fillna(' ') #Fill NaN-Values with ' ' for .lower-Function later
    Y = df.drop(columns=['id', 'message'])
    #Y = Y.astype(int) #Fill NaN-Values with ' ' for .lower-Function later
    category_names = Y.columns.tolist()
    Y = Y.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    
    return X, Y, category_names


def tokenize(text):
    # 1. Remove special characters and convert text to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())

    # 2. Split text into words (tokenize)
    tokens = word_tokenize(text)
    
    # 3. Remove stop words 
    stop_words = stopwords.words("english")
    tokens = [word for word in tokens if word not in stop_words]
    
    # 4. Lemmatization: Reduce words to their root form
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return clean_tokens


def build_model():
    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),  # Step 1: Convert text to a matrix of token counts
    ('tfidf', TfidfTransformer()),  # Step 2: Apply tf-idf transformation
    ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Step 3: Multi-output classification using RandomForest
    ])

    #parameters = {
    #'clf__estimator__n_estimators': [10, 20, 50],  # Number of trees in the Random Forest
    #'clf__estimator__min_samples_split': [2, 3],  # Minimum number of samples required to split a node
    #'clf__estimator__max_depth': [5, 10, 20]  # Maximum depth of the tree
    #}

    #new test parameters
    parameters = {
    'vect__ngram_range': [(1, 1)],  # Begrenzen auf Unigramme
    'clf__estimator__n_estimators': [10, 20],  # Weniger Bäume
    'clf__estimator__min_samples_split': [2, 3],  # Kleinere Werte
    'clf__estimator__max_depth': [5]  # Begrenzte Tiefe
    }


    model = GridSearchCV(model, param_grid=parameters, verbose=2, n_jobs=1)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    for i, category in enumerate(category_names):
        print(f"Category: {category}")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
        print("\n" + "-"*60 + "\n")

#model_filepath = saved Filename like 'model_123'
def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()