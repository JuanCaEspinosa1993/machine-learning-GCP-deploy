import pickle
import numpy as np

#Reading and consume our model and scaler binary files
url_model = "model/classifier.pickle"
url_scaler = "model/sc.pickle"

with open(url_model, 'rb') as model:
    local_classifier = pickle.load(model)

with open(url_scaler, 'rb') as scaler:
    local_scaler =  pickle.load(scaler)

#Doing predictions
new_pred = local_classifier.predict(local_scaler.transform(np.array([[40, 20000]])))
new_prob = local_classifier.predict_proba(local_scaler.transform(np.array([[40, 20000]])))[:,1]

if new_pred[0] == 0:
    print("Esta persona no comprará")
else:
    print("Esta persona comprará")
