
## UWAGA: to przyklad tylko do mnista (wiemy, ze oryg. wymiar 64 = obrazek 8x8)



from sklearn import svm
import pickle
import argparse
import os


 

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
  
 

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
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--input-dir', default="", required=True, help='input dir (default: %(default)s)')
    args = parser.parse_args()

    return  args.input_dir 

input_dir = ParseArguments()


# wczytujemy dane
x_train,y_train, x_test,y_test,classes_names = read_data(input_dir)

print("Dimension of loaded points: ", x_train.shape[1])

# osobno wczytujemy macierz przeksztalcenia
pca_object_file = open(input_dir+"/pca_object.pkl","rb")
pca = pickle.load(pca_object_file)

# tylko na x_train

x_train_inv = pca.inverse_transform(x_train)
print("Dimension after 'pca.inverse_transform': ", x_train_inv.shape[1])

print("Randomly choose 25 rows of x_train_inv, reshape each to 8x8 and display as image:")

nr_of_points = x_train_inv.shape[0]

fig, axs = plt.subplots(5, 5)

for i in np.arange(5):
    for j in np.arange(5):
        row_nr = np.random.randint(nr_of_points)
        img = x_train_inv[row_nr,:] + pca.mean_ # 64wymiarowy pkt
        img = img.reshape(8,8)
        axs[i,j].imshow(img, cmap='gray')
        axs[i,j].set_title(classes_names[y_train[row_nr]])
        axs[i,j].set_axis_off()

plt.show()
 

