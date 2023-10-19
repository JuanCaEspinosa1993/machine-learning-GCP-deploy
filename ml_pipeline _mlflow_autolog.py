import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

import mlflow
import mlflow.sklearn

#mlflow set experiment
mlflow.tracking.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(experiment_name="deploy-ml-experiment mlflow demo 4")

with mlflow.start_run(run_name="new-run1-8") as run1:
    training_data = pd.read_csv('data/storepurchasedata.csv')
    X = training_data.iloc[:, :-1].values
    y = training_data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    #Creating the object. minkowski es para la distancia euclidiana
    mlflow.log_param("no_of_neighbors", 5)
    mlflow.log_param("p", 2)
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

    #Model training
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:,1]

    cm = confusion_matrix(y_test, y_pred)
    model_accuracy = accuracy_score(y_test, y_pred)
    print(model_accuracy)
    mlflow.log_metric("accuracy", model_accuracy)
    mlflow.set_tag("classifier", "knn")

    #log model
    mlflow.sklearn.log_model(classifier, "model")
    mlflow.end_run()