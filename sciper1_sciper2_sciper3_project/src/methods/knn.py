import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind =task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        self.training_data = training_data
        self.training_labels = training_labels

        return training_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        
        test_labels = self.kNN(test_data, self.training_data, self.training_labels, self.k)

        return test_labels
    
    def euclidean_dist(example, training_examples):
        """Compute the Euclidean distance between a single example
        vector and all training_examples.

        Inputs:
            example: shape (D,)
            training_examples: shape (NxD) 
        Outputs:
            euclidean distances: shape (N,)
        """
        return np.sum((training_examples - example)**2, axis=1)**(1/2)
    
    def find_k_nearest_neighbors(k, distances):
        """ Find the indices of the k smallest distances from a list of distances.
            Tip: use np.argsort()

        Inputs:
            k: integer
            distances: shape (N,) 
        Outputs:
            indices of the k nearest neighbors: shape (k,)
        """
        
        indices = np.argsort(distances)[:k]
        return indices
    
    def predict_label(neighbor_labels):
        """Return the most frequent label in the neighbors'.

        Inputs:
            neighbor_labels: shape (N,) 
        Outputs:
            most frequent label
        """
        
        return np.argmax(np.bincount(neighbor_labels))
    
    def kNN_one_example(self, unlabeled_example, training_features, training_labels, k):
        """Returns the label of a single unlabelled example.

        Inputs:
            unlabeled_example: shape (D,) 
            training_features: shape (NxD)
            training_labels: shape (N,) 
            k: integer
        Outputs:
            predicted label
        """
        # Compute distances
        distances = self.euclidean_dist(unlabeled_example, training_features)
        
        # Find neighbors
        nn_indices = self.find_k_nearest_neighbors(k, distances)
        
        # Get neighbors' labels
        neighbor_labels = training_labels[nn_indices]
        
        # Pick the most common
        best_label = self.predict_label(neighbor_labels)
        
        return best_label
    
    def kNN(self, unlabeled, training_features, training_labels, k):
        """Return the labels vector for all unlabeled datapoints.

        Inputs:
            unlabeled: shape (MxD) 
            training_features: shape (NxD)
            training_labels: shape (N,) 
            k: integer
        Outputs:
            predicted labels: shape (M,)
        """
        return np.apply_along_axis(self.kNN_one_example, 1, self, unlabeled, training_features, training_labels, k)