from sklearn.datasets import fetch_openml
from sklearn import tree
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
mytree = tree.DecisionTreeClassifier(criterion="entropy")
mol_data = fetch_openml(name = 'pc1')
mytree.fit(mol_data.data,mol_data.target)
print(tree.export_text(mytree))
predictions = mytree.predict(mol_data.data)
pp = mytree.predict_proba(mol_data.data)
print(predictions)
metrics.accuracy_score(mol_data.target,predictions)
metrics.f1_score(mol_data.target, predictions,pos_label="true")
metrics.precision_score(mol_data.target, predictions, pos_label="true")
metrics.recall_score(mol_data.target, predictions, pos_label="true")
print(metrics.roc_auc_score(mol_data.target,pp[:,1]))
print(metrics.accuracy_score(mol_data.target,predictions))
dtc2 = tree.DecisionTreeClassifier()
cv2 = model_selection.cross_validate(dtc2,mol_data.data,mol_data.target,scoring=["accuracy","roc_auc"],cv=10,return_train_score=True)
accuracy_mean2 = cv2["train_accuracy"].mean()
test_roc_auc_mean2 = cv2["train_roc_auc"].mean()
print(test_roc_auc_mean2,accuracy_mean2)
print(accuracy_mean2,test_roc_auc_mean2)
print(cv2["test_accuracy"],cv2["test_accuracy"])
dtc = tree.DecisionTreeClassifier()
cv = model_selection.cross_validate(dtc,mol_data.data,mol_data.target,scoring=["accuracy","roc_auc"],cv=10)
accuracy_mean = cv["test_accuracy"].mean()
test_roc_auc_mean = cv["test_roc_auc"].mean()
print(accuracy_mean,test_roc_auc_mean)
print(cv["test_accuracy"],cv["test_accuracy"])





