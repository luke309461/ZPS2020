#
# mnist_create_pickles.py
#
# Przyklad: wczytuje MNIST (z sklearn.datasets) i zapisuje
# w uzywanym przez nas formacie:
# trzy pliki: train_data.pkl, test_data.pkl, class_names.pkl


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn import decomposition
 
from sklearn import datasets
from sklearn.manifold import TSNE
 
import time
import argparse

from sklearn.model_selection import train_test_split
import pickle 

# Example:
# python scripts_learn/mnist_create_pickles.py --output-dir datasets_prepared/mnist_test1/


def ParseArguments():
    parser = argparse.ArgumentParser(description="Project")
    parser.add_argument('--output-dir', default="", required=True, help='output dir (default: %(default)s)')

    args = parser.parse_args()

    return args.output_dir
        
output_dir  = ParseArguments()
    
#wczytajmy zestaw danych "digits"

literki = datasets.load_digits()
 
data = literki.data 
data_classes = literki.target 
classes_names = literki.target_names


# dzielimy zbior na treningowy i testowy

data_train, data_test, classes_train, classes_test = train_test_split(data, data_classes,
            test_size=0.2, random_state=42)




# zapisujemy dane treningowe
data_train_all_dict  =    {     'data': data_train,
                                'classes':classes_train}
            
train_data_outfile = open(output_dir + '/train_data.pkl', 'wb')
pickle.dump(data_train_all_dict, train_data_outfile)



# zapisujemy dane testowe 
data_test_all_dict  =  {'data': data_test ,
                        'classes':classes_test}
 
test_data_outfile = open(output_dir + '/test_data.pkl', 'wb')
pickle.dump(data_test_all_dict, test_data_outfile)

# zapisujemy nazwy klas
cl_names_outfile = open(output_dir + '/class_names.pkl', 'wb')
pickle.dump(classes_names, cl_names_outfile)

print("Pickles saved in ", output_dir)
 
