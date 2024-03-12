from django.shortcuts import render
from django.http import HttpResponse
from .models import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Create your views here.
homepage="index.html"
upload="upload.html"

def index(request):
    return render(request,homepage)

def upload_dataset(request):
    global data
    if request.method == 'POST':
        file = request.FILES['file']
        d = dataset(file=file)
        # d.save()
        fn = d.filename()
        print(fn)
        global data, path
        path = 'home/static/home/dataset/'+fn
        data = pd.read_csv('home/static/home/dataset/'+fn)
        datas = data.iloc[:100,:]
        table = datas.to_html()
        return render(request, 'upload.html', {'table': table})
    return render(request, 'upload.html')

def train(request):
    global data
    data = pd.read_csv('home\static\home\dataset\survey lung cancer.csv')
    data.isnull().any()
    le = LabelEncoder()
    data['LUNG_CANCER'] = le.fit_transform(data['LUNG_CANCER'])
    data['GENDER'] = le.fit_transform(data['GENDER'])
    global x_train,x_test,y_train,y_test
    if request.method == "POST":
       x=data.drop(["LUNG_CANCER"],axis=1)
       y=data['LUNG_CANCER']
       x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)
        # print(y_train)
       model = request.POST['algo']
       
       if model == "1":
           clf =DecisionTreeClassifier()
           clf.fit(x_train._get_numeric_data(), np.ravel(y_train, order='C'))
           y_pred=clf.predict(x_test)
           clf= accuracy_score(y_test,y_pred)
           msg = 'Accuracy of DecisionTreeClassifier : ' + str(clf*100)
           return render(request,'train.html',{'acc':msg})
       elif model == "2":
           lr=LogisticRegression(random_state = 0)
           lr.fit(x_train,y_train)
           y_pred=lr.predict(x_test)
           clf= accuracy_score(y_test,y_pred)
           msg = 'Accuracy of LogisticRegression : ' + str(clf*100)
           return render(request,'train.html',{'acc':msg})
       elif model == "3": 
           x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.3, random_state=30)
           rf=RandomForestClassifier(random_state=3)
           rf.fit(x_train1,y_train1)
           y_pred=rf.predict(x_test1)
           clf= accuracy_score(y_test1,y_pred)
           msg = 'Accuracy of RandomForestClassifier : ' + str(clf*100)
           return render(request,'train.html',{'acc':msg})

    return render(request,'train.html')

def predictions(request):
    global data
    if request.method == 'POST':
        d = dict(request.POST)
        # del d['csrfmiddlewaretoken']
        GENDER= float(request.POST['f1']) 
        AGE = float(request.POST['f2'])
        SMOKING = float(request.POST['f3'])
        YELLOW_FINGERS = float(request.POST['f4'])
        ANXIETY = float(request.POST['f5'])
        PEER_PRESSURE = float(request.POST['f6'])
        CHRONIC = float(request.POST['f7'])
        FATIGUE = float(request.POST['f8'])
        ALLERGY = float(request.POST['f9'])
        WHEEZING = float(request.POST['f10'])
        ALCOHOL = float(request.POST['f11'])
        COUGHING = float(request.POST['f12'])
        SHORTNESS = float(request.POST['f13'])
        SWALLOWING = float(request.POST['f14'])
        CHEST_PAIN = float(request.POST['f15'])

        PRED = [[GENDER,AGE,SMOKING,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC,FATIGUE,ALLERGY,WHEEZING,ALCOHOL,COUGHING,SHORTNESS,SWALLOWING,CHEST_PAIN]]
        print(PRED)

        clf = DecisionTreeClassifier()
        clf.fit(x_train,y_train)
        result = np.array(clf.predict(PRED))

        if result==0:
            msg1 = 'CANCER'
        else:
            msg1 = 'NO CANCER'     
        return render(request,'predictions.html',{'msg1':msg1})
    return render(request,'predictions.html')
def Accuracy(request):
    global data
    data = pd.read_csv('home\static\home\dataset\survey lung cancer.csv')
    data.isnull().any()
    le = LabelEncoder()
    data['LUNG_CANCER'] = le.fit_transform(data['LUNG_CANCER'])
    data['GENDER'] = le.fit_transform(data['GENDER'])
    x = data.drop(["LUNG_CANCER"], axis=1)
    y = data['LUNG_CANCER']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)

    classifiers = {
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'LogisticRegression': LogisticRegression(random_state=0),
        'RandomForestClassifier': RandomForestClassifier(random_state=3),
    }

    accuracy_scores = {}
    for name, clf in classifiers.items():
        clf.fit(x_train._get_numeric_data(), np.ravel(y_train, order='C'))
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores[name] = accuracy * 100

    plt.figure(figsize=(10, 6))
    plt.bar(accuracy_scores.keys(), accuracy_scores.values())
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Different Algorithms')
    plt.ylim(0, 100)
    plt.savefig('accuracy.png')

    return render(request, 'Accuracy.html')

