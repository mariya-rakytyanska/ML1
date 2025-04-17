import numpy as np
from src.utils import get_n_classes

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
        self.training_labels = np.astype(training_labels, int)
        self.number_of_classes = get_n_classes(self.training_labels)
    
        pred_labels = self.kNN(training_data)

        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """   
        test_labels = self.kNN(test_data)

        return test_labels
    
    def euclidean_dist(self, example):
        return np.sum((self.training_data - example)**2, axis=1)**(1/2)
    
    def find_k_nearest_neighbors(self, distances):
        indices = np.argsort(distances)[:self.k]
        return indices
    
    def predict_label(self, neighbor_labels):
        return np.argmax(np.bincount(neighbor_labels)) 
    
    def kNN_one_example(self, unlabeled_example):
        distances = self.euclidean_dist(unlabeled_example)
        nn_indices = self.find_k_nearest_neighbors(distances)
        neighbor_labels = self.training_labels[nn_indices]
        
        best_label = self.predict_label(neighbor_labels)
        
        return best_label
    
    def kNN(self, unlabeled):
        return np.apply_along_axis(self.kNN_one_example, 1, unlabeled)