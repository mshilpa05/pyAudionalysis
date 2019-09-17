#implementing GUI trial code

from tkinter import *
from tkinter import filedialog
from pyAudioAnalysis import audioTrainTest as aT
import numpy as np

window = Tk()

window.title("Audio Emotion Tester")

window.geometry('300x200')

OPTIONS = [
"SVM",
"KNN",
"EXTRATREES",
"GRADIENTBOOSTING",
"RANDOMFOREST"
] #etc

# default value


def clicked(classifier):
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                               filetypes=(("wav files", "*.wav"), ("all files", "*.*")))
    if classifier == "SVM":
        modelName="svmModel"
        classifier="svm"
    elif classifier == "KNN":
        modelName="knnModel"
        classifier="knn"
    elif classifier=="EXTRATREES":
        modelName = "extratreesModel"
        classifier="extratrees"
    elif classifier =="GRADIENTBOOSTING":
        modelName = "gradientboostingModel"
        classifier = "gradientboosting"
    else:
        modelName = "randomforestModel"
        classifier = "randomforest"

    print(classifier)
    print(modelName)
    isSignificant = 0.9
    Result, P, classNames = aT.fileClassification(root.filename, modelName, classifier)
    winner = np.argmax(P)
    if P[winner] > isSignificant:
        output = classNames[winner] +" probability: "+ str(P[winner])
    else:
        output =  "Can't classify sound: " + str(P)
    popupmsg(output)

def popupmsg(msg):
    popup = Tk()
    popup.wm_title("!")
    label = Label(popup, text=msg)
    label.pack(side="top", fill="x", pady=10)
    B1 = Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()
lbl1 = Label(window, text="Select Classifier")
lbl1.grid(column=5, row=0)

variable = StringVar(window)
variable.set(OPTIONS[0])
w = OptionMenu(window, variable, *OPTIONS)
w.grid(column = 20,row=0)

lbl = Label(window, text="Select Audio")
lbl.grid(column=5, row=200)

btn = Button(window, text="Browse", command=lambda:clicked(variable.get()))
btn.grid(column=20, row=200)

window.mainloop()