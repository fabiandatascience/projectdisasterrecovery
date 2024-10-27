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
    """
    Loads data from a SQLite database, preparing features and target variables for model training.

    Parameters:
    database_filepath (str): Filepath for the SQLite database containing the cleaned message and category data.

    Returns:
    X (Series): Series of messages to be used as input features.
    Y (DataFrame): DataFrame of target variables (categories) associated with each message.
    category_names (list): List of category names used as labels for the classification.

    Process:
    - Connects to the SQLite database.
    - Loads the 'messages_classified' table and extracts the 'message' column as input data (X).
    - Drops unnecessary columns and converts the remaining category columns to numeric format, storing them as target variables (Y).
    - Handles missing values by filling NaNs with 0 and converting categories to integers.

    Additional Information:
    Provides debug information on data shapes and category names.
    """
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
    """
    Processes text data by cleaning, tokenizing, removing stop words, and lemmatizing.

    Parameters:
    text (str): The input text message to be tokenized and cleaned.

    Returns:
    clean_tokens (list): List of processed words from the input text, ready for model processing.

    Process:
    - Removes special characters and converts text to lowercase.
    - Tokenizes the text into individual words.
    - Removes common stop words to retain meaningful words.
    - Lemmatizes each word to its root form for uniformity.

    Additional Information:
    This function is designed for text preprocessing in NLP tasks, making text data cleaner 
    and more informative for classification or other analysis.
    """
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
    """
    Builds and returns a machine learning pipeline with a grid search for parameter tuning.

    Returns:
    model (GridSearchCV): A GridSearchCV object containing the ML pipeline, ready for training 
    with optimized hyperparameters.

    Pipeline Structure:
    - Step 1: Text vectorization using CountVectorizer with tokenization.
    - Step 2: TF-IDF transformation for weighting.
    - Step 3: Multi-output classification using a Random Forest Classifier.

    Parameters Used in Grid Search:
    - 'vect__ngram_range': [(1, 1)] restricts vectorization to unigrams.
    - 'clf__estimator__n_estimators': [10, 20] sets a limited number of trees for Random Forest.
    - 'clf__estimator__min_samples_split': [2, 3] specifies lower split thresholds for leaf nodes.
    - 'clf__estimator__max_depth': [5] restricts tree depth for faster training.

    Additional Information:
    This function constructs a model pipeline suitable for multi-label classification 
    and optimizes it through a limited grid search to balance accuracy with computational efficiency.
    """
    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),  # Step 1: Convert text to a matrix of token counts
    ('tfidf', TfidfTransformer()),  # Step 2: Apply tf-idf transformation
    ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Step 3: Multi-output classification using RandomForest
    ])
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
    """
    Evaluates the trained model on test data and prints classification reports for each category.

    Args:
    model: The trained machine learning model.
    X_test (pd.Series or np.array): Test data, typically a series of message texts.
    Y_test (pd.DataFrame): Test labels for each category as columns.
    category_names (list): List of category names corresponding to the columns in Y_test.

    Process:
    - Generates predictions on the test data (X_test) using the provided model.
    - For each category, calculates and prints a classification report comparing actual vs. predicted labels.
    - Outputs precision, recall, and F1-score metrics for each category to assess model performance.

    Returns:
    None
    """
    Y_pred = model.predict(X_test)

    for i, category in enumerate(category_names):
        print(f"Category: {category}")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
        print("\n" + "-"*60 + "\n")

#model_filepath = saved Filename like 'model_123'
def save_model(model, model_filepath):
    """
    Saves the trained machine learning model as a pickle file.

    Args:
    model: The trained machine learning model to be saved.
    model_filepath (str): The file path where the model will be saved.

    Process:
    - Opens the specified file in write-binary mode.
    - Uses pickle to serialize and save the model, making it available for later use.
    
    Returns:
    None
    """
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