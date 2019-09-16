#implementing GUI trial code

from tkinter import *
from tkinter import filedialog
from pyAudioAnalysis import audioTrainTest as aT
import numpy as np

window = Tk()

window.title("Audio Emotion Tester")

window.geometry('500x500')

def clicked():
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                               filetypes=(("wav files", "*.wav"), ("all files", "*.*")))

    isSignificant = 0.6
    Result, P, classNames = aT.fileClassification(root.filename, "svmModel", "svm")
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

lbl = Label(window, text="Select Audio")

lbl.grid(column=0, row=0)

txt = Entry(window, width=10)

txt.grid(column=1, row=0)

btn = Button(window, text="Search", command=clicked)

btn.grid(column=2, row=0)

window.mainloop()