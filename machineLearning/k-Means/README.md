# k-Means

Naive implementation of K-Means clustering in Python. K-Means clustering is an unsupervised machine learning algorithm that aims to partition N observations into K clusters in which each observation belongs to the cluster with the nearest mean.

Here's an animation of some random samples of data being clustered into k=3 clusters:

![kmeans_anim](https://user-images.githubusercontent.com/8327505/227825237-47f5f137-15a1-436d-b3b6-7943998401c6.gif)


Step 1 - Set k random centroids, one for each cluster

Step 2 - Compute the euclidean distance from each data point to each cluster centroid,
		 assign data point to closest cluster
		 
Step 3 - Set new clusters centroids by computing each cluster's mean x and y

Step 4 - For each cluster, measure the error between the old centroids and the new centroids

Step 5 - If the total error is above a threshold, go to Step 2. Otherwise, end the algorithm.
