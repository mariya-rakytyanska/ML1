import numpy as np
import itertools


class KMeans(object):
    """
    kNN classifier object.
    """

    def __init__(self, max_iters=500):
        """
        Call set_arguments function of this class.
        """
        self.max_iters = max_iters
        self.centroids = None
        self.best_permutation = None

    def init_centers(self,data, K):
        self.centroids = data[np.random.choice(data.shape[0], K, replace=False)]
        return self.centroids
    
    def compute_centers(self, data, cluster_assignments, K):
        self.centroids = np.zeros((K, data.shape[1]), dtype=float)
    
        for i in range(K):
            rows = data[cluster_assignments == i]
            if rows.size > 0:
                self.centroids[i] = rows.mean(axis = 0)

        return self.centroids

    def k_means(self,data, K, max_iters):
        self.centroids = self.init_centers(self,data, K)

        for i in range(max_iters):
            old_centers = self.centroids.copy()  

            cluster_assignments = self.find_closest_cluster(self,self.compute_distance(self,data, self.centroids))
            self.centroids = self.compute_centers(self,data, cluster_assignments, K)
        
            if np.all(self.centroids == old_centers): 
                break
        cluster_assignments = self.find_closest_cluster(self,self.compute_distance(self,data, self.centroids))
        return self.centroids, cluster_assignments
    
    def compute_distance(self, data):
        self.centroids = np.sqrt(((data[:, np.newaxis, :] - self.centroids[np.newaxis, : , :])**2).sum(axis = 2))
        return self.centroids
    
    def find_closest_cluster(distances):
        cluster_assignments = np.argmin(distances,axis = 1)
        return cluster_assignments
    
    def assign_labels_to_centers(self, centers, cluster_assignments, true_labels):
        K = np.shape(centers)[0]
        self.assign_labels_to_centers = np.zeros((K,),dtype=np.float64)
    
        for i in range(K):
            rows = true_labels[cluster_assignments == i]
            self.assign_labels_to_centers[i] = np.argmax(np.bincount(rows))
        
        return self.best_permutation

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.
        Hint:
            (1) Since Kmeans is unsupervised clustering, we don't need the labels for training. But you may want to use it to determine the number of clusters.
            (2) Kmeans is sensitive to initialization. You can try multiple random initializations when using this classifier.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,).
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        K = np.shape(self.centers)[0]
        distances = self.compute_distance(self,training_data)
        cluster_center_label = self.assign_labels_to_centers(self, self.centroids, self.find_closest_cluster(distances), training_labels)
    
        for i in range(K):
            rows = training_labels[self.find_closest_cluster(distances) == i]
            cluster_center_label[i] = np.argmax(np.bincount(rows))
        

        pred_labels = self.predict(self, training_data)

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        K = np.shape(self.centers)[0]
        test_labels = self.k_means(self,test_data, K, self.max_iters)[1]
        return test_labels
