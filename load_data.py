import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def loadData(folder, feature_type):
    data = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if(feature_type+"_2.json"==filename):
                
                name = root.split(os.sep)[-3]
                filepath = root+os.sep+filename

                d = pd.read_json(filepath)
                d = pd.DataFrame(StandardScaler().fit_transform(d), columns=d.columns)
                d = pd.DataFrame(PCA(n_components=2).fit_transform(d))

                nrows = d.shape[0]
                n = int(nrows/2)

                labels = pd.DataFrame([0]*n + [1]*(nrows-n), columns=["label"])
                d = pd.concat([d,labels], axis=1)
                
                names = pd.DataFrame([name]*nrows, columns=["name"])
                d = pd.concat([d,names], axis=1)

                data.append(d)
    return(pd.concat(data))

data = loadData("data/large_scale/","Features")

data