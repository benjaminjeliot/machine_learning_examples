
# coding: utf-8

# # Iris K-Means clustering example
# 
# Simple example of a K-Means clustering using the Iris dataset.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.cluster


# Load the iris data.

# In[2]:


iris = sklearn.datasets.load_iris()


# Plot the iris data, colored by variety.

# In[3]:


fig, ax = plt.subplots()
for i in range(3):
    ax.scatter(
        iris.data[np.where(iris.target == i), 0],
        iris.data[np.where(iris.target == i), 1],
        label=iris.target_names[i])
ax.legend()
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
plt.show()


# Run the clustering algorithm.

# In[4]:


centroid, label, inertia = sklearn.cluster.k_means(X=iris.data, n_clusters=3)


# Relabel to be consistent with correct labels.

# In[5]:


relabeled = np.zeros(label.shape)
for i in range(3):
    indices = np.where(label == i)
    new_label = np.round(np.mean(iris.target[indices])).astype(np.int)
    relabeled[indices] = new_label


# Plot the clusters, colored by cluster.

# In[6]:


fig, ax = plt.subplots()
for i in range(3):
    ax.scatter(
        iris.data[np.where(relabeled == i), 0],
        iris.data[np.where(relabeled == i), 1],
        label=iris.target_names[i])
ax.legend()
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
ax.scatter(centroid[:, 0], centroid[:, 1], c='black', marker='X', s=200)
plt.show()


# The clustering does a pretty good job of matching the correct labels! :-)
