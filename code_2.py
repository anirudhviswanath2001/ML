from sklearn import datasets , tree, model_selection
dtc = tree.DecisionTreeClassifier()
tuned_dtc = model_selection.GridSearchCV(dtc, parameters, scoring = "roc_auc", cv=5 )
cv = model_selection.cross_validate(tuned_dtc, dia.data,dia.target, scoring = ["roc_auc", "accuracy"], cv =10)
cv 