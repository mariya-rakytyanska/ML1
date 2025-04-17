import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters


    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        training_labels = np.astype(training_labels, int)
        self.W = self.logistic_regression_train_multi(training_data, label_to_onehot(training_labels, get_n_classes(training_labels)))
        pred_labels = LogisticRegression.logistic_regression_predict_multi(training_data, self.W)

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        pred_labels = LogisticRegression.logistic_regression_predict_multi(test_data, self.W)

        return pred_labels

    def f_softmax(data, W):
        x = data @ W
        x = x - x.max(axis=1, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=1, keepdims=True)
    
    def loss_logistic_multi(self, data, labels, w):
        return -np.sum(np.sum(labels * np.log(self.f_softmax(data, w))))
    
    def gradient_logistic_multi(data, labels, W):
        return data.T @ (LogisticRegression.f_softmax(data, W) - labels)
    
    def logistic_regression_predict_multi(data, W):
        return np.apply_along_axis(np.argmax, 1, LogisticRegression.f_softmax(data, W))

    def logistic_regression_train_multi(self, data, labels):
        D = data.shape[1] 
        C = labels.shape[1] 
        weights = np.random.normal(0, 0.1, (D, C))
        for it in range(self.max_iters):
            gradient = LogisticRegression.gradient_logistic_multi(data, labels, weights)
            weights = weights - (self.lr * gradient)

            predictions = LogisticRegression.logistic_regression_predict_multi(data, weights)
            if accuracy_fn(predictions, onehot_to_label(labels)) == 100:
                print(it)
                break
            
        return weights
