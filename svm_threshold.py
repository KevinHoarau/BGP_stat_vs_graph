from load_data import loadData

import argparse
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, GridSearchCV

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def applyML(data, n_splits=4, verbose=False, threshold=10):
    
    names = data["name"].to_numpy()
    Y = data["label"].to_numpy()
    
    X = data.drop(columns=["name","label"]).to_numpy()
    
    results_train = []
    results_test = []

    group_kfold = LeaveOneGroupOut()
    i = 0
    for train_index, test_index in group_kfold.split(X, Y, names):
        i += 1

        if(verbose):
            print("===========================================")
            print("Iteration %i:" % (i))
            print("  Train set:", np.unique(names[train_index]))
            print("  Test set: ", np.unique(names[test_index]))
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        names_train, names_test = names[train_index], names[test_index]
        
        models = {
        "SVM_rbf":
            (
                SVC(random_state=1),
                {"SVM_rbf__gamma" : [0.1,0.01], "SVM_rbf__C" : [100,10], "SVM_rbf__kernel" : ['rbf'], "SVM_rbf__tol": [1e-3]}
            )
        }
        
        r_train = {}
        r_test = {}
        
        for (name, (model, params)) in models.items():
            
            pipe = Pipeline([(name, model)])
            cv = GroupKFold(n_splits=3).split(X_train, y_train, names_train)
            search = GridSearchCV(pipe, params, cv=cv, scoring='accuracy', n_jobs=16).fit(X_train, y_train)
            clf = search.best_estimator_
            
            if(verbose):
                print("  Best params (%s): %s" % (name, str(search.best_params_).replace(",",",\n"+" "*21)))
            
            pred_test = []
            true_test = []
            
            pred_train = []
            true_train = []
            
            for n in np.unique(names_test):
                mask_0 = np.logical_and(names_test==n,y_test==0)
                mask_1 = np.logical_and(names_test==n,y_test==1)
                
                pred_test += [int(clf.predict(X_test[mask_0]).sum()>threshold),int(clf.predict(X_test[mask_1]).sum()>threshold)]
                true_test += [0,1]
                
            for n in np.unique(names_train):
                mask_0 = np.logical_and(names_train==n,y_train==0)
                mask_1 = np.logical_and(names_train==n,y_train==1)
                
                pred_train += [int(clf.predict(X_train[mask_0]).sum()>threshold),int(clf.predict(X_train[mask_1]).sum()>threshold)]
                true_train += [0,1]
                
            r_train[name] = accuracy_score(true_train, pred_train)
            r_test[name] = accuracy_score(true_test, pred_test)
        
        if(verbose):
            print("  ----------------------------------------")
            print("  |    Model    | Train acc. | Test acc. |")
            print("  ----------------------------------------")
        
            for (name, (model, params)) in models.items():
                print("  | %10s  |    %.2f    |   %.2f    |" % (name, r_train[name] , r_test[name]))
                print("  ----------------------------------------")
            
        results_train.append(r_train)
        results_test.append(r_test)
        
    results_train = pd.DataFrame(results_train)
    results_test = pd.DataFrame(results_test)
    
    return(results_train, results_test)

parser = argparse.ArgumentParser(description='Evaluate ML models')
parser.add_argument('anomalies',type=str, choices=['LS', 'OH', 'PH'],
                    help='type of anomalies: large scale, origin hijacking, path_hijacking')

parser.add_argument('features',type=str, choices=['S', 'G', 'GK'],
                    help='type of features: statistical features, graph features, graph features with kcore')

parser.add_argument('threshold',type=int,metavar="[0-30]",
                    help='threshold')

args = parser.parse_args()

anomalies = {'LS':"large_scale", 'OH':"origin_hijacking", 'PH':"path_hijacking"}
features = {'S':"Features", 'G':"GraphFeatures_allNodes", 'GK':"GraphFeatures"}

print(args.anomalies)
print(args.features)

data = loadData("data/"+anomalies[args.anomalies]+"/",features[args.features])
results_train, results_test = applyML(data, verbose=True, threshold=args.threshold)

print("===========================================")
print("Results:")

print(results_test.mean())