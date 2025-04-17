import numpy as np
import itertools
from ..utils import get_n_classes


class KMeans(object):
    """
    kNN classifier object.
    """

    def __init__(self, max_iters=500, K=10):
        """
        Call set_arguments function of this class.
        """
        self.max_iters = max_iters
        self.centroids = None
        self.best_permutation = None
        self.K = K

    def init_centers(self,data):
        self.centroids = data[np.random.choice(data.shape[0], self.K, replace=False)]
        return 0
    
    def compute_centers(self, data, cluster_assignments):
        self.centroids = np.zeros((self.K, data.shape[1]), dtype=float)
    
        for i in range(self.K):
            rows = data[cluster_assignments == i]
            if rows.size > 0:
                self.centroids[i] = rows.mean(axis = 0)
        return self.centroids

    def k_means(self, data):
        self.init_centers(data)

        for i in range(self.max_iters):
            cluster_assignments = KMeans.find_closest_cluster(self.compute_distance(data))
            old_centroids = self.centroids.copy()
            self.centroids = self.compute_centers(data, cluster_assignments)
        
            if np.all(self.centroids == old_centroids): 
                print(i)
                break
            
        cluster_assignments = KMeans.find_closest_cluster(self.compute_distance(data))
        return self.centroids, cluster_assignments
    
    def compute_distance(self, data):
        N = data.shape[0]
        K = self.centroids.shape[0]

        distances = np.zeros((N, K))
        for k in range(K):
            center = self.centroids[k]
            distances[:, k] = np.sqrt(((data - center) ** 2).sum(axis=1))
        return distances
    
    def find_closest_cluster(distances):
        cluster_assignments = np.argmin(distances,axis = 1)
        return cluster_assignments
    
    def assign_labels_to_centers(self, centers, cluster_assignments, true_labels):
        K = np.shape(centers)[0]
        self.best_permutation = np.zeros((self.K,),dtype=np.float64)
    
        for i in range(K):
            rows = true_labels[cluster_assignments == i]
            self.best_permutation[i] = np.argmax(np.bincount(rows))
        
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
        training_labels = np.astype(training_labels,int)
    
        final_centers, cluster_assignments = self.k_means(training_data)
        self.assign_labels_to_centers(final_centers, cluster_assignments, training_labels)
        pred_labels = self.predict(training_data)
        
        return pred_labels
        
    

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        test_labels = np.zeros((np.shape(test_data)[0],))
        center_ass = KMeans.find_closest_cluster(self.compute_distance(test_data))
    
        for i in range(np.shape(test_data)[0]):
            test_labels[i] = self.best_permutation[center_ass[i]]
        
        return test_labels
