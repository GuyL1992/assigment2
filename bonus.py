from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


data = load_iris(return_X_y=True, as_frame=True)[0]
interias = []

for i in range (2,11):
    model = KMeans(i, random_state=0)
    interias.append(model.fit(data).inertia_)

x_axis = np.array([i for i in range (2,11)])
y_axis = np.array(interias)

plt.plot(x_axis,y_axis,color = "green", markevery = [1], marker='$◌$', markersize=20)

plt.annotate("Elbow Point",(3,interias[1]),  xycoords='data',
            xytext=(0.6,0.6), textcoords="axes fraction",
            arrowprops=dict(width = 0.001, facecolor='black',),
            horizontalalignment='right', verticalalignment='top',
            )

plt.xlabel('K - Values')
plt.ylabel('Interias')
plt.title("Elbow method for selection of Optimal K - clusters")
plt.savefig('elbow.png')

