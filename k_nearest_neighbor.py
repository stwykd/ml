import numpy as np

class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        return self.predict_labels(self.compute_distances(X), k=k)

    def compute_distances(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train.
        """
        return ((X**2).sum(axis=1, keepdims=True) + (self.X_train**2).sum(axis=1) +
                -2 * X.dot(self.X_train.T))**.5

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # Contains labels of the k nearest neighbors for each ith test point.
            closest_y = self.y_train[np.argsort(dists[i, :])[:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred


from cifar10_utils import *
X_train, y_train, X_test, y_test = load_CIFAR10()
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print("Loaded and unrolled training set and test set from CIFAR10")

num_training, num_test = 5000, 500
train_mask = np.random.randint(X_train.shape[0], size=num_training)
test_mask = np.random.randint(X_test.shape[0], size=num_test)
X_train, y_train = X_train[train_mask], y_train[train_mask]
X_test, y_test = X_test[test_mask], y_test[test_mask]
print("Subsampled training set and test set")

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
print("Trained knn classifier")

def cross_validate_k():
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train.reshape(-1, 1), num_folds)
    k_to_accuracies = {}

    [k_to_accuracies.setdefault(k, []) for k in k_choices]
    for i in range(num_folds):
        classifier = KNearestNeighbor()
        X_val_train = np.vstack(X_train_folds[0:i] + X_train_folds[i+1:])
        y_val_train = np.vstack(y_train_folds[0:i] + y_train_folds[i+1:])[:,0]
        classifier.train(X_val_train, y_val_train)
        for k in k_choices:
            y_val_pred = classifier.predict(X_train_folds[i], k=k)
            num_correct = np.sum(y_val_pred == y_train_folds[i][:,0])
            accuracy = float(num_correct) / len(y_val_pred)
            k_to_accuracies[k] = k_to_accuracies[k] + [accuracy]
    return max(k_to_accuracies, key=lambda x: np.mean(k_to_accuracies[x]))

chosen_k = cross_validate_k()
print("Found the best value for k through cross validation")

y_test_pred = classifier.predict(X_test, k=chosen_k)
print("Predicted values for the test set")

num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' %
      (num_correct, num_test, accuracy))
