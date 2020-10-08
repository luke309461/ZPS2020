#
# prepare_data_divorce
#
# Przyklad: z katalogu --input-dir
# 
# trzy pliki: train_data.pkl, test_data.pkl, class_names.pkl
#
# Wylicza macierz PCA na punktach train_data.pkl i je przeksztalca
# To samo przeksztalcenie wykonuje na train_data.pkl
# Wynik zapisywany jest do --output-dir


from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
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

# Example:
# python scripts_learn/prepare_data_divorce.py --input-dir datasets/divorce/ --output-dir datasets_prepared/divorce_ver1

 


def save_data(x_train , y_train, x_test , y_test, classes_names, output_dir):
    

    # zapisujemy dane treningowe
    x_train_all_dict  =    {     'data': x_train ,
                                    'classes':y_train}
                
    train_data_outfile = open(output_dir + '/train_data.pkl', 'wb')
    pickle.dump(x_train_all_dict, train_data_outfile)



    # zapisujemy dane testowe 
    x_test_all_dict  =  {'data': x_test  ,
                        'classes':y_test}
     
    test_data_outfile = open(output_dir + '/test_data.pkl', 'wb')
    pickle.dump(x_test_all_dict, test_data_outfile)

    # zapisujemy nazwy klas
    cl_names_outfile = open(output_dir + '/class_names.pkl', 'wb')
    pickle.dump(classes_names, cl_names_outfile)

    print("Pickles saved in ", output_dir)
    
    


def ParseArguments():
    parser = argparse.ArgumentParser(description="Project")
    parser.add_argument('--input-dir', default="", required=True, help='data dir (default: %(default)s)')
    parser.add_argument('--output-dir', default="", required=True, help='output dir (default: %(default)s)')
    parser.add_argument('--fraction', default="0.25", required=False, help='size of test set (fration) (default: %(default)s)')
    args = parser.parse_args()

    return args.input_dir, args.output_dir, args.fraction
        
input_dir, output_dir, test_size_fraction   = ParseArguments()

test_size_fraction = float(test_size_fraction)

# wczytaj plik csv, osobno jest train i test (pomijamy validation), osobno lables
df = pd.read_csv(input_dir+"/madelon_train.data", sep=" ")


# Zamieniamy DataFrame (df) na macierz numpy

data_all = df.to_numpy()


# w ostatniej kolumnie jakies "smieci" pomijamy
x_train = data_all[:,:499] 

# osobno wczutujemy labels:
df = pd.read_csv(input_dir+"/madelon_train.labels", sep=" ")

y_train = df.to_numpy().reshape(-1)

 
# nazwy klas -- damy tutaj wszystkie unikalne numery, ktore wystepuja w data_classes
classes_names = np.unique(y_train)


# To samo ze zbiorem testowym:
df = pd.read_csv(input_dir+"/madelon_valid.data", sep=" ")
data_all = df.to_numpy()
x_test = data_all[:,:499] 

df = pd.read_csv(input_dir+"/madelon_valid.labels", sep=" ")
y_test = df.to_numpy().reshape(-1)

 

  

save_data(x_train , y_train, x_test , y_test, classes_names, output_dir)
    
         


 
 
