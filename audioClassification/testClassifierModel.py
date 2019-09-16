#!/usr/local/bin/python2
from pyAudioAnalysis import audioTrainTest as aT
from sys import argv
import numpy as np
from flask import Flask
from flask import jsonify
from flask import request

app = Flask(__name__)

# script, filename = argv
# isSignificant = 0.6  # try different values.
#
# # P: list of probabilities
# Result, P, classNames = aT.fileClassification(filename, "svmModel", "svm")
# winner = np.argmax(P)  # pick the result with the highest probability value.
#
# # is the highest value found above the isSignificant threshhold?
# if P[winner] > isSignificant:
#     print("File: " + filename + " is in category: " + classNames[winner] + ", with probability: " + str(P[winner]))
# else:
#     print("Can't classify sound: " + str(P))


@app.route('/test/SVM', methods=['POST'])
def testSVM():
    isSignificant = 0.6
    filename = request.json['fileName']
    Result, P, classNames = aT.fileClassification(filename, "svmModel", "svm")
    winner = np.argmax(P)
    if P[winner] > isSignificant:
        return "File: " + filename + " is in category: " + classNames[winner] + ", with probability: " + str(P[winner])
    else:
        return "Can't classify sound: " + str(P)

@app.route('/test/KNN', methods=['POST'])
def testKNN():
    isSignificant = 0.6
    filename = request.json['fileName']
    Result, P, classNames = aT.fileClassification(filename, "knnModel", "knn")
    winner = np.argmax(P)
    if P[winner] > isSignificant:
        return "File: " + filename + " is in category: " + classNames[winner] + ", with probability: " + str(P[winner])
    else:
        return "Can't classify sound: " + str(P)

@app.route('/test/extraTrees', methods=['POST'])
def testExtraTrees():
    isSignificant = 0.6
    filename = request.json['fileName']
    Result, P, classNames = aT.fileClassification(filename, "extratreesModel", "extratrees")
    winner = np.argmax(P)
    if P[winner] > isSignificant:
        return "File: " + filename + " is in category: " + classNames[winner] + ", with probability: " + str(P[winner])
    else:
        return "Can't classify sound: " + str(P)


@app.route('/test/gradientBoosting', methods=['POST'])
def testGradientBoosting():
    isSignificant = 0.6
    filename = request.json['fileName']
    Result, P, classNames = aT.fileClassification(filename, "gradientboostingModel", "gradientboosting")
    winner = np.argmax(P)
    if P[winner] > isSignificant:
        return "File: " + filename + " is in category: " + classNames[winner] + ", with probability: " + str(P[winner])
    else:
        return "Can't classify sound: " + str(P)

@app.route('/test/randomForest', methods=['POST'])
def testRandomForest():
    isSignificant = 0.6
    filename = request.json['fileName']
    Result, P, classNames = aT.fileClassification(filename, "randomforestModel", "randomforest")
    winner = np.argmax(P)
    if P[winner] > isSignificant:
        return "File: " + filename + " is in category: " + classNames[winner] + ", with probability: " + str(P[winner])
    else:
        return "Can't classify sound: " + str(P)

app.run(host='0.0.0.0', port=5000)