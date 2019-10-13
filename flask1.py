from flask import  Flask, render_template, url_for
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.preprocessing import StandardScaler

dirs = "/home/sambhav/Desktop/flask-co2/static/data"

PEOPLE_FOLDER = os.path.join('static', 'people_photo')

#ML model
def predict():

    for cv in os.listdir(dirs) :
        path = os.path.join(dirs, cv)
        df = pd.read_csv(path)
        x = np.array(df["year"]).reshape(-1,1)
        y = np.array(df["emissions"]).reshape(-1,1)
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
        
        reg = LinearRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)
        prediction = list(pred)
        plt.plot(x_test, pred)
        plt.scatter(x_test, y_test)
        fig = plt.show()

        return fig


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER


@app.route("/")
@app.route("/home")
def hello():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/map")
def map():
    return render_template('map.html')

@app.route("/demographic")
def demographic():
    return render_template('demographic.html')

@app.route("/predict")
def predict():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], '1.jpg')
    table_img = os.path.join(app.config['UPLOAD_FOLDER'], 'Capture.PNG')

    return render_template("predict.html", user_image = full_filename, table_img = table_img)

if __name__ == '__main__':
    app.run(debug=True)