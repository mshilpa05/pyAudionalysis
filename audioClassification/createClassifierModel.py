 #!/usr/local/bin/python2
from pyAudioAnalysis import audioTrainTest as aT
import os
from sys import argv
script, dirname = argv
print ('asdfghjkl;',argv)

subdirectories = os.listdir(dirname)[:14]

print (subdirectories)

subdirectories = [dirname + "/" + subDirName for subDirName in subdirectories]
print(subdirectories)
#pyAudioAnalysis has 5 classifiers:
#   1)SVM
#   2)KNN
#   3)randomForest
#   4)GradientBoosting

#   5)Extra Trees

#SVM classifier used:
aT.featureAndTrain(subdirectories, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmModel", False)

#KNN classifier used
aT.featureAndTrain(subdirectories, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knnModel", False)

#Extra Trees Model used
aT.featureAndTrain(subdirectories, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "extratrees", "extratreesModel", False)

#Gradient Boosting classifier used
aT.featureAndTrain(subdirectories, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "gradientboosting", "gradientboostingModel", False)

#RandomForest classifier used
aT.featureAndTrain(subdirectories, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "randomforest", "randomforestModel", False)





