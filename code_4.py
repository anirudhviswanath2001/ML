from sklearn import tree
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import metrics 
from sklearn import model_selection

#importing dataset 45019: BioResponse
dia = datasets.fetch_openml(name = 'pc1')
#setting up parameters
parameters = [1,2,5,10,20,140]



train_scores = []
test_scores = []

for min_sample_leaf in parameters:
    model = tree.DecisionTreeClassifier(min_samples_leaf=min_sample_leaf)
    cv = model_selection.cross_validate(model,dia.data,dia.target,scoring="roc_auc",cv=10,return_train_score=True)
    train_score = cv["train_score"].mean()
    train_scores.append(train_score)


    #evaluating model on test data:
    model.fit(dia.data,dia.target)
    pp = model.predict_proba(dia.data)
    test_score = metrics.roc_auc_score(dia.target, pp[:,1])
    test_scores.append(test_score)

print(train_scores,"\n")
print(test_scores,"\n")


plt.plot(parameters,train_scores,label = "train_scores")
plt.plot(parameters,test_scores,label = "test_scores")
plt.savefig('my_plot1.png')
