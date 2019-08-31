 #!/usr/local/bin/python2
from pyAudioAnalysis import audioTrainTest as aT
import os
from sys import argv
script, dirname = argv
print ('asdfghjkl;',argv)

subdirectories = os.listdir(dirname)[:5]

print (subdirectories)

subdirectories = [dirname + "/" + subDirName for subDirName in subdirectories]
print(subdirectories)

aT.featureAndTrain(subdirectories, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmModel", False)