#
# Axes3D, example 
#

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd


fig = plt.figure()
#to bedzie 3d
ax = fig.add_subplot(111, projection='3d')

#ile punktow
n = 50
 
 
for c, m  in [('r', 'o' ), ('b', '^' )]:
    xs = rnd.rand(n)*20-15
    ys = rnd.rand(n)*30-20
    zs = rnd.rand(n)*40-20
    ax.scatter(xs, ys, zs, c=c, marker=m)
 
plt.title('Losowe punkty 3d')
plt.show()


