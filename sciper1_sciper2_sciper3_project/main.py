import argparse

import numpy as np

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.knn import KNN
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data

    # EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        #feature_data = np.load("features.npz", allow_pickle=True)
        feature_data = np.load(os.path.join(args.data_path, "features.npz"), allow_pickle=True)
        xtrain, xtest = feature_data["xtrain"], feature_data["xtest"]
        ytrain, ytest = feature_data["ytrain"], feature_data["ytest"]
        xtrain = np.astype(xtrain, int)
        xtest = np.astype(xtest, int)
        ytrain = np.astype(ytrain, int)
        ytest = np.astype(ytest, int)

    # ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path, "dog-small-64")
        xtrain, xtest, ytrain, ytest = load_data(data_dir)

    ## 2. Then we must prepare it. This is where you can create a validation set, normalize, add bias, etc.
    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        ### WRITE YOUR CODE HERE
        training_samples = np.shape(xtrain)[0]
        xvalidation = xtrain[:training_samples//10]
        yvalidation = ytrain[:training_samples//10]
        xtrain = xtrain[training_samples//10:]
        ytrain = ytrain[training_samples//10:]

        val_mean = np.apply_along_axis(np.mean, 0, xvalidation)
        val_std = np.apply_along_axis(np.std, 0, xvalidation)
        xvalidation = normalize_fn(xvalidation, val_mean, val_std)
        xvalidation = append_bias_term(xvalidation)
        pass

    ### WRITE YOUR CODE HERE to do any other data processing

    train_mean = np.apply_along_axis(np.mean, 0, xtrain)
    test_mean = np.apply_along_axis(np.mean, 0, xtest)
    train_std = np.apply_along_axis(np.std, 0, xtrain)
    test_std = np.apply_along_axis(np.std, 0, xtest)
    xtrain = normalize_fn(xtrain, train_mean, train_std)
    xtest = normalize_fn(xtest, test_mean, test_std)

    xtrain = append_bias_term(xtrain)
    xtest = append_bias_term(xtest)

    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "knn":
        arr = np.zeros(shape = (50,))
        for i in range(1, 51):
            validate = KNN(i)
            validate.fit(xtrain, ytrain)
            prediction = validate.predict(xvalidation)
            acc = accuracy_fn(prediction, yvalidation)
            arr[i-1] = acc

        K = np.argmax(arr) + 1

        method_obj = KNN(K)  ### WRITE YOUR CODE HERE
        pass

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(args.lr, args.max_iters) ### WRITE YOUR CODE HERE
        pass

    elif args.method == "kmeans":
        method_obj = KMeans(args.max_iters)  ### WRITE YOUR CODE HERE
        pass

    ## 4. Train and evaluate the method
    # Fit (:=train) the method on the training data for classification task
    xtrain = np.concatenate((xvalidation, xtrain))
    ytrain = np.concatenate((yvalidation, ytrain))
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)
    preds_val = method_obj.predict(xvalidation)

    # Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds_val, yvalidation)
    macrof1 = macrof1_fn(preds_val, yvalidation)
    print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / knn / logistic_regression / kmeans / nn (MS2)",
    )
    parser.add_argument(
        "--data_path", default="data", type=str, help="path to your dataset"
    )
    parser.add_argument(
        "--data_type", default="features", type=str, help="features/original(MS2)"
    )
    parser.add_argument(
        "--K", type=int, default=1, help="number of neighboring datapoints used for knn"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, otherwise use a validation set",
    )

    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument(
        "--nn_type",
        default="cnn",
        help="which network to use, can be 'Transformer' or 'cnn'",
    )
    parser.add_argument(
        "--nn_batch_size", type=int, default=64, help="batch size for NN training"
    )

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
