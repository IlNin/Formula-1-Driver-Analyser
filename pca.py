from flask import Flask, render_template, redirect, url_for, request, jsonify
from flask import make_response
from flask_cors import CORS, cross_origin

import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from matplotlib import pyplot as plt

app = Flask(__name__)
#app.config['CORS_HEADERS'] = 'Access-Control-Allow-Origin'

CORS(app)
#CORS(app, resources={r"/pca": {"origins": "http://localhost:8000"}})

@app.route("/")
def home():
    return "hi"
@app.route("/index")

@app.route('/pca', methods=['POST'])
#@cross_origin(origin='localhost',headers=['Access-Control-Allow-Origin', 'Content-Type'])
def pca():
   if request.method == 'POST':
        # Reads and parses the data
        print("POST request acknowledged")
        data = request.get_json(force=True)
        print("Data read and parsed!")
        
        # Prepares the data for the PCA process
        dataForPCA = []
        for element in data['value']:
            championships = element['championships']
            points = element['points']
            poles = element['poles']
            wins = element['wins']
            dataForPCA.append([championships, points, poles, wins])
        print("Tuples for the PCA created!")
        
        # Transforms the data
        d_std = preprocessing.StandardScaler().fit_transform(dataForPCA)
        print("Tuples pre-processed!")
        
        # Chooses number of PCA components
        pca = PCA(n_components=2)

        # Applies PCA
        dpca = pca.fit_transform(d_std)    
        print("PCA applied on the tuples!")
        
        # Plots the data
        showData = 0
        if (showData):
            plt.plot(
                dpca[:,0], 
                dpca[:,0],
                'o',
                markersize = 7,
                color = 'green',
                alpha = 0.5,
                label = "PCA"
            )
            plt.xlabel("Y1")
            plt.ylabel("Y2")
            plt.legend()
            plt.show()
           
        # Sends the new data back
        response = jsonify(dpca.tolist())
        response.headers.add('Access-Control-Allow-Origin', '*')
        print("Sending back PCA data!")
        return response

if __name__ == "__main__":
    app.run(debug=True)