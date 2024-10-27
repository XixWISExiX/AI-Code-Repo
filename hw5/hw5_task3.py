#
# Template for Task 3: Kmeans Clustering
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Your Task --- #
# import libraries as needed 
# --- end of task --- #

# -------------------------------------
# load data 
# note we do not need label 
data = np.loadtxt('../data_sets/crimerate.csv', delimiter=',')
[n,p] = np.shape(data)
sample = data[:,0:-1]
# -------------------------------------

# --- Your Task --- #
# pick a proper number of clusters 
k = 3
# --- end of task --- #


# --- Your Task --- #
# implement the Kmeans clustering algorithm 
# you need to first randomly initialize k cluster centers 
random_indices = np.random.choice(n, k, replace=False)
centroids = sample[random_indices,:]  # Randomly selected k centroids
     
# Set initial values for loop
previous_centroids = centroids.copy()
tolerance = 1e-4
max_iterations = 100
converged = False
iteration = 0


# then start a loop 
while not converged and iteration < max_iterations:
     # Init the label array (clusters) or y
     label_cluster = []

     # Go through all points in the data set and assign them to the closest centroid
     for sample_point in sample:
          # Compute distances from current point to all centroids
          distances = np.linalg.norm(sample_point - centroids, axis=1)

          # Assign the point to the nearest centroid
          label_cluster.append(np.argmin(distances))
    
     label_cluster = np.array(label_cluster)  # Convert to a NumPy array

     # Go through the clusters and update the centriod locations by taking the mean.
     for i in range(k):
          # Get all points assigned to cluster i
          cluster_points = sample[label_cluster == i]

          # Compute new centroid as the mean of the cluster points
          if len(cluster_points) > 0:
               centroids[i, :] = np.mean(cluster_points, axis=0)
    
    # Check for convergence (super small changes are not needed)
     centroid_shift = np.linalg.norm(centroids - previous_centroids)
     if centroid_shift < tolerance:
          converged = True
     else:
          previous_centroids = centroids.copy()

     iteration += 1


# when clustering is done, 
# store the clustering label in `label_cluster' 
# cluster index starts from 0 e.g., 
# label_cluster[0] = 1 means the 1st point assigned to cluster 1
# label_cluster[1] = 0 means the 2nd point assigned to cluster 0
# label_cluster[2] = 2 means the 3rd point assigned to cluster 2
# --- end of task --- #


# # the following code plot your clustering result in a 2D space
# pca = PCA(n_components=2)
# pca.fit(sample)
# sample_pca = pca.transform(sample)
# idx = []
# colors = ['blue','red','green','m']

# for i in range(k):
#      idx = np.where(label_cluster == i)
#      plt.scatter(sample_pca[idx,0], sample_pca[idx,1] ,color=colors[i],facecolors='none')
# plt.xlabel('Average x1')
# plt.ylabel('Average x2')
# plt.title('2D Plot of K-Means Clustering Algorithm')
# plt.show()

# the following code plot your clustering result in a 3D space
pca = PCA(n_components=3)
pca.fit(sample)
sample_pca = pca.transform(sample)
idx = []
colors = ['blue','red','green','m']

fig = plt.figure()
ax = plt.axes(projection='3d')

for i in range(k):
     idx = np.where(label_cluster == i)
     ax.scatter(sample_pca[idx,0], sample_pca[idx,1], sample_pca[idx,2], color=colors[i], facecolors='none')
ax.set_xlabel('Average x1')
ax.set_ylabel('Average x2')
ax.set_zlabel('Average x3')
plt.title('3D Plot of K-Means Clustering Algorithm')
plt.show()

print("For this algorithm implementation of K-Means I used numpy for the computation of euclidean distance.")
print("In this case, being in 100D, which saves time without a python for loop.")
print("Along with other additions to simply code output and time as well (like mp.mean).")
print("There is also a threashold to ignore super small changes to not go through all iterations, so that is more of an optimization change that doesn't affect the core algorithm.")

print("For k=2 in 2D, we can see a divide between the two groups from the middle and this sort of grouping seems to be fine, but seems like a higher k value could be more beneficial.")

print("For k=3 in 2D, we can see a divide between the three groups, one in the middle like before but also one for the top section which is more scattered out and seems that it does a better job at describing the feature distribution than k=2.")

print("For k=3 in 3D, we can see that the green values also have a higher x3 axis value than we could previously see in the 2D plot. So we can further back our assumption that k=3 is more informative than a k=2 plot would be because we wouldn't have gotten much more information from that extra axis in the k=2 case.")