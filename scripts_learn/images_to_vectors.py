import numpy as np
import sys
import argparse

import glob, os

#from sklearn.datasets import fetch_20newsgroups
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import numpy as np
from sklearn import metrics

import matplotlib.pyplot as plt
import imageio

from skimage.color import rgb2gray
from skimage.transform import rescale, resize 

import pickle

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
    



def read_images(path, classes_names, im_height, im_width, convert_to_gray):
    print("classes_names = ", classes_names)
    X = []
    Y = []
    for classs in classes_names:
        # search for all files in path
        for file_name in list((glob.glob(path + classs + "/**"))):
            print("klasa = ", classs, ", file = ", file_name)
            
            
            im=imageio.imread(file_name)
            
            # jesli (..,..,4) -- tzn. ze obrazek przezroczysty
            # ignorujemy 4ta warstwe:
            
            if(im.shape[2]>3):
                im=im[:,:,:3]
                
            
            if(convert_to_gray=="yes"):
                im=rgb2gray(im)
                
            # zmieniamy rozmiar obrazka na staly
            
            im=resize(im,(im_width, im_height))
                
            im_vec = im.reshape(-1) # reshape to one long vector
            
            X.append(im_vec)
            Y.append(classes_dict[classs])

    print("len(X) = ", len(X))
    X=np.vstack(X) # robimy macierz liczba obrazkow x im_width*im_height (*3 jesli rgb)
    Y=np.vstack(Y).reshape(-1) # wektor
    
    return X, Y



def ParseArguments():
    parser = argparse.ArgumentParser(description="Project")
    parser.add_argument('--input-dir', default="", required=True, help='input dir (default: %(default)s)')
    parser.add_argument('--output-dir', default="", required=True, help='output dir (default: %(default)s)')
    parser.add_argument('--h', default="200", required=False, help='height of resulting img (default: %(default)s)')
    parser.add_argument('--w', default="200", required=False, help='width of resulting img (default: %(default)s)')
    parser.add_argument('--conv', default="yes", required=False, help='convert to gray or not (default: %(default)s)')
    args = parser.parse_args()

    return args.input_dir,args.output_dir, args.h, args.w, args.conv 
        
input_dir, output_dir, img_height, img_width, conv_to_grayscale_bool   = ParseArguments()

img_height=int(img_height)
img_width=int(img_width)


# main program
 

print("input-dir = ", input_dir)

# folder data_dir/train should contain only subfolders = classes

# first, we read only subfolders names in   data_dir/train
classes_names = []

classes_dict = {}

counter = 0;

for file in glob.glob(input_dir + "/train/**"):
    tmp = file.rsplit('/', 3)
    classes_names.append(tmp[len(tmp) - 1])
    classes_dict[tmp[len(tmp) - 1]] = counter;
    counter = counter + 1;

print("Klasy  = ", classes_names)

# read in train data

x_train , y_train = read_images(input_dir + "/train/", classes_names, img_height, img_width ,conv_to_grayscale_bool)

x_test , y_test  = read_images(input_dir + "/test/", classes_names, img_height, img_width ,conv_to_grayscale_bool)


save_data(x_train , y_train, x_test , y_test, classes_names, output_dir)
    
     
quit()    

# BoW
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train_data)

# classifier
clf = MultinomialNB(alpha=.01)
clf.fit(X_train, Y_train)

# read in test data
X_test_data, Y_test = read_data(data_dir + "/test/", classes_names)
X_test = vectorizer.transform(X_test_data)

Y_pred = clf.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(Y_test, Y_pred))

