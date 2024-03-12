from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd



data = pd.read_csv(r'C:\Users\hyder\OneDrive\Desktop\CODE\survey lung cancer.csv')
data.isnull().any()
le = LabelEncoder()
data['LUNG_CANCER'] = le.fit_transform(data['LUNG_CANCER'])
data['GENDER'] = le.fit_transform(data['GENDER'])
global x_train,x_test,y_train,y_test
x=data.drop(["LUNG_CANCER"],axis=1)
y=data['LUNG_CANCER']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=30)

rf=RandomForestClassifier(random_state=3)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
clf= accuracy_score(y_test,y_pred)
msg = 'Accuracy of RandomForestClassifier : ' + str(clf*100)
print(msg)