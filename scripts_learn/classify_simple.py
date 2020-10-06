#
# classify_simple.py
#
# Przyklad: z katalogu --input_dir
# trzy pliki: train_data.pkl, test_data.pkl, class_names.pkl
#
# Uczy sie na train_data.pkl, przewiduje (dwa proste kl) na test_data.pkl


# Example
# python scripts_learn/classify_simple.py --input-dir datasets_prepared/mnist_test1_pca3d/


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn import decomposition
 
from sklearn import datasets, metrics 
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
 
import time
import argparse

from sklearn.model_selection import train_test_split
import pickle 

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

 
def read_data(input_dir):
    # wczytujemy dane treningowe:
    train_data_infile = open(input_dir + '/train_data.pkl', 'rb')  # czytanie z pliku
    data_train_all_dict =  pickle.load(train_data_infile)

    x_train = data_train_all_dict["data"]
    y_train = data_train_all_dict["classes"]

    # wczytujemy dane testowe:
    test_data_infile = open(input_dir + '/test_data.pkl', 'rb')  # czytanie z pliku
    data_test_all_dict =  pickle.load(test_data_infile)

    x_test= data_test_all_dict["data"]

    y_test = data_test_all_dict["classes"]

    # i nazwy klas 
    cl_names_infile = open(input_dir + '/class_names.pkl', 'rb')
    classes_names =  pickle.load(cl_names_infile)

    print("Data loaded from " + input_dir)
    
    return x_train, y_train, x_test, y_test, classes_names
 
def ParseArguments():
    parser = argparse.ArgumentParser(description="Project")
    parser.add_argument('--input-dir', default="", required=True, help='input dir (default: %(default)s)')
    parser.add_argument('--k', default="3", required=False, help='k = nr of neighb. (default: %(default)s)')
    args = parser.parse_args()

    return args.input_dir, args.k
        
input_dir, k_nr_neighbours  = ParseArguments()
    
k_nr_neighbours=int(k_nr_neighbours)
    

x_train,y_train, x_test,y_test,classes_names = read_data(input_dir)

    


### KLASYFIKACJA

## knn 
# definiujemy klasyfikator (3ciu najbl. sasiadow)
knn_clf = KNeighborsClassifier(n_neighbors=k_nr_neighbours)

# "uczymy" sie na zbiorze treningowym
start_time = time.time()
print("Learning and predicting with knn ...",  end =" ")
knn_clf.fit(x_train, y_train)

# przewidujemy na testowym
y_pred = knn_clf.predict(x_test)
print("  took %s seconds " % round((time.time() - start_time),5))
    
# na testowym znalismy prawdziwe klasy, mozemy porownac jak "dobrze" poszlo
# (rozne metryki, tutaj przyklad

metric_accuracy = metrics.accuracy_score(y_test,y_pred)
print("knn: accuracy = ", metric_accuracy)


# dla binarnych:
# metric_f1 = metrics.f1_score(y_test,y_pred)
# metric_precision = metrics.precision_score(y_test,y_pred)
# metric_recall = metrics.recall_score(y_test,y_pred)


print("full classification report:")
if type(classes_names) is not list:
    target_nms = classes_names.astype(str)
else:
    target_nms = classes_names
    
print(classification_report(y_test, y_pred, target_names = target_nms))     
  
# support vector machine, rozne gamma:  
    
for gamma in np.arange(1,10)/500:    
    svm_clf = SVC(gamma=gamma)
    print("Learning ...", "svm(gamma=", gamma, ")...",  end =" ")
    svm_clf.fit(x_train, y_train)
    print("  took %s seconds " % round((time.time() - start_time),5))

    
    y_pred = svm_clf.predict(x_test)
    metric_accuracy = metrics.accuracy_score(y_test,y_pred)
    print("svm(gamma=", gamma, "): accuracy = ", metric_accuracy)


 
