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
        return 0
    
    def compute_centers(self, data, cluster_assignments, K):
        self.centroids = np.zeros((K, data.shape[1]), dtype=float)
    
        for i in range(K):
            rows = data[cluster_assignments == i]
            if rows.size > 0:
                self.centroids[i] = rows.mean(axis = 0)

        return self.centroids

    def k_means(self, data, K):
        self.init_centers(data, K)

        for i in range(self.max_iters):
            cluster_assignments = KMeans.find_closest_cluster(self.compute_distance(data))
            self.centroids = self.compute_centers(data, cluster_assignments, K)
        
            if np.all(self.centroids == self.centroids): 
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
        self.best_permutation = np.zeros((K,),dtype=np.float64)
    
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
        
            
        arr = np.zeros(shape = (50,))
        for i in range(1, 51):
            final_centers, cluster_assignments = self.k_means(training_data, i)
            self.assign_labels_to_centers(final_centers, cluster_assignments, training_labels)
            pred_labels = KMeans.predict(self,training_data)
            acc = 100*np.sum(np.equal(pred_labels, training_labels))/np.size(training_labels)
            arr[i-1] = acc

        K = np.argmax(arr) + 1
        final_centers, cluster_assignments = self.k_means(training_data, K)
        self.assign_labels_to_centers(final_centers, cluster_assignments, training_labels)
        pred_labels = KMeans.predict(self,training_data)
        
        
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
