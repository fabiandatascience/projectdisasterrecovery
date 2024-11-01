import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_classified', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message'] 
    genre_names = list(genre_counts.index) 
    
    # Calculate the top categories by message count
    top_categories = df.iloc[:, 4:].sum().sort_values(ascending=False).head(10)
    top_category_names = top_categories.index
    top_category_values = top_categories.values

    # Calculate the proportion of "aid_related" messages
    aid_related_counts = df['aid_related'].value_counts()
    aid_related_labels = ['Aid Related', 'Not Aid Related']
    aid_related_values = aid_related_counts.values

    # Number of categories per message (shows how many categories each message triggers)
    categories_per_message = df.iloc[:, 4:].sum(axis=1).value_counts().sort_index()
    categories_per_message_x = categories_per_message.index
    categories_per_message_y = categories_per_message.values
    
    # create visuals
    graphs = [
        # Genre Distribution
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        
        # Top 10 Categories
        {
            'data': [
                Bar(
                    x=top_category_names,
                    y=top_category_values
                )
            ],
            'layout': {
                'title': 'Top 10 Categories by Message Count',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category"}
            }
        },

        # Aid Related Distribution
        {
            'data': [
                Pie(
                    labels=aid_related_labels,
                    values=aid_related_values
                )
            ],
            'layout': {
                'title': 'Proportion of "Aid Related" Messages'
            }
        },

        # Distribution of Number of Categories per Message
        {
            'data': [
                Bar(
                    x=categories_per_message_x,
                    y=categories_per_message_y
                )
            ],
            'layout': {
                'title': 'Distribution of Number of Categories per Message',
                'yaxis': {'title': "Number of Messages"},
                'xaxis': {'title': "Number of Categories"}
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    print("Prediction result:", model.predict([query]))

    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
