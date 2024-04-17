import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, redirect, url_for, request, render_template
import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.pipeline import Pipeline
import os
import sqlite3

app = Flask(__name__)

@app.route('/input/<name>')
def display_name(name):
    return render_template('result2.html', output=name)

def get_prediction_proba(docx):
    df = pd.read_csv('dataset.csv', names=['text', 'emotion'])
    df['Clean_Text'] = df['text'].apply(nfx.remove_userhandles)
    df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)
    Xfeatures = df['Clean_Text']
    Ylabels = df['emotion']

    x_train, x_test, y_train, y_test = train_test_split(Xfeatures, Ylabels, test_size=0.3, random_state=42)

    pipe_lr = Pipeline(steps=[('cv', CountVectorizer()), ('lr', LogisticRegression(solver='lbfgs', max_iter=200))])
    pipe_lr.fit(x_train, y_train)
    pipe_nb = Pipeline(steps=[('cv', CountVectorizer()), ('nb', naive_bayes.MultinomialNB(alpha=0.3))])
    pipe_nb.fit(x_train, y_train)
    n_estimators = 10
    pipe_svm = Pipeline(steps=[('cv', CountVectorizer()), ('svm', BaggingClassifier(svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=-1))])
    pipe_svm.fit(x_train, y_train)
    pipe_rf = Pipeline(steps=[('cv', CountVectorizer()), ('rf', RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42))])
    pipe_rf.fit(x_train, y_train)

    result_lr = pipe_lr.predict_proba([docx])
    result_nb = pipe_nb.predict_proba([docx])
    result_svm = pipe_svm.predict_proba([docx])
    result_rf = pipe_rf.predict_proba([docx])

    max_lr = np.max(pipe_lr.predict_proba([docx]))
    max_nb = np.max(pipe_nb.predict_proba([docx]))
    max_svm = np.max(pipe_svm.predict_proba([docx]))
    max_rf = np.max(pipe_rf.predict_proba([docx]))
    if(max_lr > max_nb and max_lr > max_svm and max_lr > max_rf):
        return [result_lr, pipe_lr.predict([docx])[0]]
    elif(max_nb > max_lr and max_nb > max_svm and max_nb > max_rf):
        return [result_nb, pipe_nb.predict([docx])[0]]
    elif(max_svm > max_lr and max_svm > max_nb and max_svm > max_rf):
        return [result_svm, pipe_svm.predict([docx])[0]]
    else:
        return [result_rf, pipe_rf.predict([docx])[0]]

@app.route('/TEXT_EMOTION_DETECTOR', methods=['POST', 'GET'])
def hello_world():
    if request.method == 'POST':
        f = open("database.txt", "a")
        user = request.form['text']
        f.write("Text ->  " + user + "\n")
        ex1 = user
        probability = get_prediction_proba(ex1)
        predicted_emotion = probability[1]
        probability = probability[0][0]
        for i in range(len(probability)):
            probability[i] = probability[i] * 100
        savefile = 'static/graph.png'
        if os.path.exists(savefile):
            os.remove(savefile)
            plt.clf()
        emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
        plt.bar(range(len(emotions)), probability, tick_label=emotions, color=['red', 'green', 'yellow', 'hotpink', 'lightgrey', 'orange'])
        plt.xlabel('Emotions')
        plt.ylabel('Confidence Level (%)')
        plt.title('Predicted Probability')
        plt.xticks(rotation=45)
        plt.savefig(savefile)
        confidence = np.max(probability)
        f.write("Predicted emotion ->  " + str(predicted_emotion) + "\n")
        f.write("Confidence rate -> " + str(confidence) + "\n\n")
        sentence = str(predicted_emotion) + " With confidence: {:.2f}".format(confidence) + "%"
        f.close()
        return redirect(url_for('display_name', name=sentence))

@app.route('/details')
def graph():
    return("""
        <head>
        <link rel="icon" type="image/x-icon" href="logo.jpeg" />
        <title>DETAILS ABOUT PREDICTED EMOTION</title>
        <style>
        body {
            background-image: url('https://i.pinimg.com/originals/b3/99/62/b399626d1f40e2342b2d85e1eebf2722.gif');
            background-repeat: no-repeat;
            background-size: cover;
            }
        h1 {
            font-family: verdana;
            color: white;
            font-size: 60px;
            text-align: center;
            }
        p {
            font-family: arial;
            font-size: 24px;
            color: white;
            }
        </style>
        </head>
        <body>
        <h1>DETAILED GRAPH OF PREDICTED EMOTION</h1>
        <p>This graph shows the predicted probability of various emotions based on the given text.</p>
        <p><img src="/static/graph.png" alt="Graph"></p>
        </body>
    """)

if __name__ == '__main__':
    app.run(debug=True)