import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
import pickle

training_data = pd.read_csv('data/storepurchasedata.csv')
#print(training_data.describe())

X = training_data.iloc[:, :-1].values
y = training_data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

"""
WE are using a KNN classifier in this example
*n_neighbors = 5, -*Number of neighbors
*metric = 'minkowski', p = 2  -For Euclidian distance claculation
"""
#Creating the object
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

#Model training
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)[:,1]

cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))

#Doing a new predcition

new_prediction = classifier.predict(sc.transform(np.array([[40,20000]])))
new_probability = classifier.predict_proba(sc.transform(np.array([[40,20000]])))[:,1]

new_pred = classifier.predict(sc.transform(np.array([[42,50000]])))
new_prob = classifier.predict_proba(sc.transform(np.array([[42,50000]])))[:,1]

# 0 He wil buy   1 He will not buy
print(f"con una edad de 40 años y un salario de 20000: ", new_prediction)
print(f"La probabildad de compra es: ", new_probability)
print("=="*10)
print(f"con una edad de 42 años y un salario de 50000: ", new_pred)
print(f"La probabildad de compra es: ", new_prob)

model_file = "model/classifier.pkl"
pickle.dump(classifier, open(model_file, 'wb'))

scaler_file = "model/sc.pkl"
pickle.dump(sc, open(scaler_file, 'wb'))