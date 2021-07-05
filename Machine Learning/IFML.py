'''
Most of this code is adapted from "Unsupervised Motion Artifact Detection in Wrist-Measured EDA"
Focuses on in sample and out of sample prediction for Isolation Forest
'''

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut,GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier,IsolationForest
from sklearn.neural_network import MLPClassifier
import joblib




nJobs = 12  # Number of cores to use

# Load feature matrices, labels, and groups (denoting which labeled time
# segment each row of the feature matrix comes from)

featuresAll = np.loadtxt('JohnAll.csv',delimiter=',')
featuresAcc = np.loadtxt('JohnAcc.csv',delimiter=',')
featuresEda = np.loadtxt('JohnEda.csv',delimiter=',')
labels = np.loadtxt('JohnLabels.csv',delimiter=',')
groups = np.loadtxt('JohnGroups.csv',delimiter=',')
NickAll = np.loadtxt('NickAll.csv',delimiter=',')
NickAcc = np.loadtxt('NickAcc.csv',delimiter=',')
NickEda = np.loadtxt('NickEda.csv',delimiter=',')
NickLabels = np.loadtxt('NickLabels.csv',delimiter=',')
NickGroups = np.loadtxt('NickGroups.csv',delimiter=',')

# Leave-one-group-out cross-validation
cv = LeaveOneGroupOut()
# Isolation Forest
# Parameter tuning by grid search
IFparameters = {'n_estimators': 10*np.arange(1,21)}

IFgsAll = GridSearchCV(IsolationForest(),
                     IFparameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=True,
                     verbose=1)
IFgsAll.fit(featuresAll,1-labels,groups)
bestNumTreesAll = IFgsAll.best_params_['n_estimators']

IFgsAcc = GridSearchCV(IsolationForest(),
                     IFparameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=True,
                     verbose=1)
IFgsAcc.fit(featuresAcc,1-labels,groups)
bestNumTreesAcc = IFgsAcc.best_params_['n_estimators']

IFgsEda = GridSearchCV(IsolationForest(),
                     IFparameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=True,
                     verbose=1)
IFgsEda.fit(featuresEda,1-labels,groups)
bestNumTreesEda = IFgsEda.best_params_['n_estimators']


IFpredAll = np.zeros(np.shape(labels))
IFpredAcc = np.zeros(np.shape(labels))
IFpredEda = np.zeros(np.shape(labels))
IFpredNickAll = np.zeros(np.shape(NickLabels))
IFpredNickAcc = np.zeros(np.shape(NickLabels))
IFpredNickEda = np.zeros(np.shape(NickLabels))

for train, test in cv.split(featuresAll,labels,groups):
    IFAll = IsolationForest(n_estimators=bestNumTreesAll)
    IFAll.fit(featuresAll[train,:])
    IFpredAll[test] = IFAll.decision_function(featuresAll[test,:])

    IFAcc = IsolationForest(n_estimators=bestNumTreesAcc)
    IFAcc.fit(featuresAcc[train,:])
    IFpredAcc[test] = IFAcc.decision_function(featuresAcc[test,:])

    IFEda = IsolationForest(n_estimators=bestNumTreesEda)
    IFEda.fit(featuresEda[train,:])
    IFpredEda[test] = IFEda.decision_function(featuresEda[test,:])



print('IF AUC ALL: %f (%s)' % (roc_auc_score(1-labels,IFpredAll),IFgsAll.best_params_))
print('IF AUC ALL: %f (%s)' % (roc_auc_score(1-labels,IFpredAcc),IFgsAcc.best_params_))
print('IF AUC ALL: %f (%s)' % (roc_auc_score(1-labels,IFpredEda),IFgsEda.best_params_))


#joblib.dump(IFgsAll, 'Isolation Forest')

#model = joblib.load(model_file_path)




## Isolation Forest
# Perform grid search cross-validation on UTD data to identify best parameters
IFparameters = {'n_estimators': 10*np.arange(1,21)}

IFGsAll = GridSearchCV(IsolationForest(),
                        IFparameters,'roc_auc',n_jobs=12,cv=cv,refit=False,
                        verbose=1)
IFGsAcc = GridSearchCV(IsolationForest(),
                        IFparameters,'roc_auc',n_jobs=12,cv=cv,refit=False,
                        verbose=1)
IFGsEda = GridSearchCV(IsolationForest(),
                        IFparameters,'roc_auc',n_jobs=12,cv=cv,refit=False,
                        verbose=1)



IFAll = IsolationForest(n_estimators=bestNumTreesAll)
IFAll.fit(featuresAll)
IFNickPredAll = IFAll.decision_function(NickAll)

IFAcc = IsolationForest(n_estimators=bestNumTreesAcc)
IFAcc.fit(featuresAcc)
IFNickPredAcc = IFAcc.decision_function(NickAcc)

IFEda = IsolationForest(n_estimators=bestNumTreesEda)
IFEda.fit(featuresEda)
IFNickPredEda = IFEda.decision_function(NickEda)

#out of sample prediction
print('IF AUC ALL PREDICTION: %f (%s)' % (roc_auc_score(1-NickLabels,IFNickPredAll),
                            IFgsAll.best_params_))
print('IF AUC ACC PREDICTION: %f (%s)' % (roc_auc_score(1-NickLabels,IFNickPredAcc),
                            IFgsAcc.best_params_))
print('IF AUC EDA PREDICTION: %f (%s)' % (roc_auc_score(1-NickLabels,IFNickPredEda),
                            IFgsEda.best_params_))

IFgsAll2 = GridSearchCV(IsolationForest(),
                     IFparameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=True,
                     verbose=1)
IFgsAll2.fit(NickAll,1-NickLabels,NickGroups)
bestNumTreesAll2 = IFgsAll2.best_params_['n_estimators']

IFgsAcc2 = GridSearchCV(IsolationForest(),
                     IFparameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=True,
                     verbose=1)
IFgsAcc2.fit(NickAcc,1-NickLabels,NickGroups)
bestNumTreesAcc2 = IFgsAcc2.best_params_['n_estimators']

IFgsEda2 = GridSearchCV(IsolationForest(),
                     IFparameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=True,
                     verbose=1)
IFgsEda2.fit(NickEda,1-NickLabels,NickGroups)
bestNumTreesEda2 = IFgsEda2.best_params_['n_estimators']

cv = LeaveOneGroupOut()
IFpredAll2 = np.zeros(np.shape(labels))
IFpredAcc2 = np.zeros(np.shape(labels))
IFpredEda2 = np.zeros(np.shape(labels))
IFpredNickAll2 = np.zeros(np.shape(NickLabels))
IFpredNickAcc2 = np.zeros(np.shape(NickLabels))
IFpredNickEda2 = np.zeros(np.shape(NickLabels))

for train, test in cv.split(NickAll,NickLabels,NickGroups):
    IFAll2 = IsolationForest(n_estimators=bestNumTreesAll2)
    IFAll2.fit(NickAll[train,:])
    IFpredAll2[test] = IFAll2.decision_function(NickAll[test,:])

    IFAcc2 = IsolationForest(n_estimators=bestNumTreesAcc2)
    IFAcc2.fit(NickAcc[train,:])
    IFpredAcc2[test] = IFAcc2.decision_function(NickAcc[test,:])

    IFEda2 = IsolationForest(n_estimators=bestNumTreesEda2)
    IFEda2.fit(NickEda[train,:])
    IFpredEda2[test] = IFEda2.decision_function(NickEda[test,:])

print('IF AUC ALL: %f (%s)' % (roc_auc_score(1-NickLabels,IFpredAll2),IFgsAll2.best_params_))
print('IF AUC ALL: %f (%s)' % (roc_auc_score(1-NickLabels,IFpredAll2),IFgsAll2.best_params_))
print('IF AUC ALL: %f (%s)' % (roc_auc_score(1-NickLabels,IFpredAll2),IFgsAll2.best_params_))

IFGsAll2 = GridSearchCV(IsolationForest(),
                        IFparameters,'roc_auc',n_jobs=12,cv=cv,refit=False,
                        verbose=1)
IFGsAcc2 = GridSearchCV(IsolationForest(),
                        IFparameters,'roc_auc',n_jobs=12,cv=cv,refit=False,
                        verbose=1)
IFGsEda2 = GridSearchCV(IsolationForest(),
                        IFparameters,'roc_auc',n_jobs=12,cv=cv,refit=False,
                        verbose=1)



IFAll2 = IsolationForest(n_estimators=bestNumTreesAll2)
IFAll2.fit(NickAll)
IFNickPredAll2 = IFAll2.decision_function(featuresAll)

IFAcc2 = IsolationForest(n_estimators=bestNumTreesAcc2)
IFAcc2.fit(NickAcc)
IFNickPredAcc2 = IFAcc2.decision_function(featuresAcc)

IFEda2 = IsolationForest(n_estimators=bestNumTreesEda2)
IFEda2.fit(NickEda)
IFNickPredEda2 = IFEda2.decision_function(featuresEda)

#out of sample prediction
print('IF AUC ALL PREDICTION: %f (%s)' % (roc_auc_score(1-labels,IFNickPredAll2),
                            IFgsAll2.best_params_))
print('IF AUC ACC PREDICTION: %f (%s)' % (roc_auc_score(1-labels,IFNickPredAcc2),
                            IFgsAcc2.best_params_))
print('IF AUC EDA PREDICTION: %f (%s)' % (roc_auc_score(1-labels,IFNickPredEda2),
                            IFgsEda2.best_params_))