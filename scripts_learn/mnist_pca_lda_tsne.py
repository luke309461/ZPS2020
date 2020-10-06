#
# tsne_example.py
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn import decomposition
#or from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from metric_learn import NCA
import time


ile=1000

#wczytajmy zestaw danych "digits"

literki = datasets.load_digits()
#literki = datasets.load_wine()

# ~ >>> literki = datasets.load_digits()
# ~ >>> points = literki.data[:ile]

# ~ >>> data_classes = literki.target[:ile]
# ~ >>> pp=points[1,:]
# ~ >>> plt.imshow(np.reshape(pp,(8,8))
# ~ >>> plt.show()



#wezmy tylko 'ile' punktow (wszystkich jest 115008)
points = literki.data[:ile]
data_classes = literki.target[:ile]
classes_names = literki.target_names

print("points.shape=",points.shape) 
print("Calculating PCA...", end="", flush=True)
start_time = time.time()
pca = decomposition.PCA(n_components=3) 
pca.fit(points)
points_pca_reduced = pca.transform(points)
print("took %s seconds" % (time.time() - start_time))
# ~ print("PCA ok.")
# ~ print("points.shape_reduced=",points_reduced.shape)
 
 
 
# Compute ICA
print("Calculating ICA...", end="", flush=True)
start_time = time.time()
ica = decomposition.FastICA(n_components=3)
points_ica_reduced= ica.fit_transform(points)   
print("took %s seconds " % (time.time() - start_time))


print("Calculating kernel PCA...", end="", flush=True)
start_time = time.time()
kpca = decomposition.KernelPCA(kernel="poly",n_components=3, gamma=70)
points_kpca_reduced=points;
kpca.fit(points_kpca_reduced)
points_kpca_reduced = kpca.transform(points_kpca_reduced)
print("took %s seconds " % (time.time() - start_time))

print("Calculating LDA...", end="", flush=True)
start_time = time.time()
lda=LinearDiscriminantAnalysis(n_components=3)
#points_lda_reduced = lda.fit(points,data_classes).transform(points)
points_lda_reduced = lda.fit_transform(points,data_classes)

print("took %s seconds " % (time.time() - start_time))

# ~ #NCA
# ~ print("Calculating NCA...", end="", flush=True)
# ~ start_time = time.time()
# ~ # nca = NCA(max_iter=1000, learning_rate=0.01)
# ~ # nca.num_dims=3
# ~ # nca.fit(points,data_classes)
# ~ # points_nca_reduced = nca.transform(points);
# ~ points_nca_reduced=points_pca_reduced
# ~ print("took %s seconds" % (time.time() - start_time))



print("Calculating t-SNE...", end="", flush=True)
start_time = time.time()
tsne = TSNE(n_components=3, random_state=0)
points_tsne_reduced = tsne.fit_transform(points)
print("\t\t took %s seconds" % round((time.time() - start_time)))



fig_pca = plt.figure(1)
ax_pca = fig_pca.add_subplot(111, projection='3d')


fig_lda = plt.figure(2)
ax_lda = fig_lda.add_subplot(111, projection='3d')



fig_ica = plt.figure(3)
ax_ica = fig_ica.add_subplot(111, projection='3d')



fig_kpca = plt.figure(4)
ax_kpca = fig_kpca.add_subplot(111, projection='3d')


# ~ fig_nca = plt.figure(5)
# ~ ax_nca = fig_nca.add_subplot(111, projection='3d')


fig_tsne = plt.figure(6)
ax_tsne = fig_tsne.add_subplot(111, projection='3d')


 
import numpy.random as rnd
 
for wt in range(0,data_classes.max()+1):
    points_pca=points_pca_reduced[data_classes == wt];    
    ax_pca.scatter(points_pca[:,0], points_pca[:,1], points_pca[:,2], label=classes_names[wt])
    
    points_lda=points_lda_reduced[data_classes == wt];    
    ax_lda.scatter(points_lda[:,0], points_lda[:,1], points_lda[:,2], label=classes_names[wt])
    
    

    
    points_ica=points_ica_reduced[data_classes == wt];    
    ax_ica.scatter(points_ica[:,0], points_ica[:,1], points_ica[:,2], label=classes_names[wt])
    
    
    
    points_kpca=points_kpca_reduced[data_classes == wt];    
    ax_kpca.scatter(points_kpca[:,0], points_kpca[:,1], points_kpca[:,2], label=classes_names[wt])

    
    # ~ points_nca=points_nca_reduced[data_classes == wt];
    # ~ ax_nca.scatter(points_nca[:,0], points_nca[:,1], points_nca[:,2], label=classes_names[wt])

    points_tsne=points_tsne_reduced[data_classes == wt];
    ax_tsne.scatter(points_tsne[:,0], points_tsne[:,1], points_tsne[:,2], label=classes_names[wt])

    
#print(xs_lda)
    
    
  
ax_pca.set_title("PCA")
ax_lda.set_title("LDA")
ax_ica.set_title("ICA")
ax_kpca.set_title("Kernel PCA")
#ax_nca.set_title("NCA")
ax_tsne.set_title("t-SNE")



    
    
    
 
ax_pca.legend()
ax_lda.legend()
ax_ica.legend()
ax_kpca.legend()
#ax_nca.legend()
ax_tsne.legend()


plt.show();

quit()

print("Kliknij w obrazek..")
plt.waitforbuttonpress()
 
 
print("Liczymy t-SNE ...")
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
tsne = TSNE(n_components=3, random_state=0)
points_reduced_tsne = tsne.fit_transform(points)
print("t-SNE ok.")

print("points_reduced_tsne=",points_reduced_tsne.shape)

for wt in range(0,data_classes.max()+1):
    points_tmp=points_reduced_tsne[data_classes == wt];
    xs = points_tmp[:,0]
    ys = points_tmp[:,1]
    zs = points_tmp[:,2]
    ax2.scatter(xs, ys, zs, label=classes_names[wt])

plt.title('t-SNE')
ax2.legend()


plt.show()
 
