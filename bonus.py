from turtle import color
from matplotlib import colors
import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


data_set = load_iris(return_X_y=True, as_frame=True)[0]
interias = []

for i in range (1,11):
    model = KMeans(i, random_state=0)
    interias.append(model.fit(data_set).inertia_)

x_axis = np.array([i for i in range (1,11)])
y_axis = np.array(interias)

plt.plot(x_axis,y_axis,color = "green", markevery = [1], marker='$◌$', markersize=20)
plt.annotate("elbow",(2,interias[1]), xytext=(0.4, 0.4), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="->"))

plt.xlabel('K - Values')
plt.ylabel('Interias')
plt.title("Elbow method for selection of Optimal K - clusters")
plt.savefig('elbow.png')

