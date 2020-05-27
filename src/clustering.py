import numpy as np
from scipy import stats
import math

class KMeans():
    """
    KMeans. Class for building an unsupervised clustering model
    """

    def __init__(self, k, max_iter=20):

        """
        :param k: the number of clusters
        :param max_iter: maximum number of iterations
        """

        self.k = k
        self.max_iter = max_iter
        self.centers = None

    def init_center(self, x):
        """
        initializes the center of the clusters using the given input
        :param x: input of shape (n, m)
        :return: updates the self.centers
        """

        ################################
        #      YOUR CODE GOES HERE     #
        ################################

        self.centers = np.zeros((self.k, x.shape[1]))
        indices = np.random.choice(np.arange(x.shape[0]), self.k, replace=False)
        self.centers = x[indices]

    def revise_centers(self, x, labels):
        """
        it updates the centers based on the labels
        :param x: the input data of (n, m)
        :param labels: the labels of (n, ). Each labels[i] is the cluster index of sample x[i]
        :return: updates the self.centers
        """

        for i in range(self.k):
            wherei = np.squeeze(np.argwhere(labels == i), axis=1)
            self.centers[i, :] = x[wherei, :].mean(0)

    def predict(self, x):
        """
        returns the labels of the input x based on the current self.centers
        :param x: input of (n, m)
        :return: labels of (n,). Each labels[i] is the cluster index for sample x[i]
        """
        ##################################
        #      YOUR CODE GOES HERE       #
        ##################################
        labels = np.zeros((x.shape[0]), dtype=int)
        values = np.zeros((x.shape[0], 2))
        values[:, 1] = math.inf

        for i in range(self.k):
            center = self.centers[i]
            new_vals = (x - center) ** 2
            new_vals = np.sum(new_vals, axis=1)
            new_vals = np.sqrt(new_vals)
            np.reshape(new_vals, (new_vals.shape[0], 1))
            new_vals = np.vstack((np.ones(new_vals.shape[0]) * i, new_vals)).T
            values[:, 0] = np.where(values[:, 1] > new_vals[:, 1], new_vals[:, 0], values[:, 0])
            values[:, 1] = np.where(values[:, 1] > new_vals[:, 1], new_vals[:, 1], values[:, 1])

        labels = values[:, 0]
        labels = labels.astype('int')


        return labels

    def get_sse(self, x, labels):
        """
        for a given input x and its cluster labels, it computes the sse with respect to self.centers
        :param x:  input of (n, m)
        :param labels: label of (n,)
        :return: float scalar of sse
        """
        ##################################
        #      YOUR CODE GOES HERE       #
        ##################################
        sse = 0.
        for i in range(x.shape[0]):
            err = np.linalg.norm(x[i] - self.centers[labels[i]])
            sse += err ** 2

        return sse

    def get_purity(self, x, y):
        """
        computes the purity of the labels (predictions) given on x by the model
        :param x: the input of (n, m)
        :param y: the ground truth class labels
        :return:
        """
        ##################################
        #      YOUR CODE GOES HERE       #
        ##################################
        labels = self.predict(x)
        correct = 0
        for i in range(self.k):
            predict_k_at = self.get_array_idxs(labels, i)
            y_preds = y[predict_k_at]
            y_list = y_preds.tolist()
            max_class = max(set(y_list), key=y_list.count)
            correct += y_list.count(max_class)

        purity = correct / y.shape[0]

        return purity

    @staticmethod
    def get_array_idxs(a, val):
        """
        this function returns an np array of indexes where a == val.
        :param a: input data of (n)
        :return: indexes where a[idx] == val
        """
        b = a == val
        r = np.array(range(len(b)))

        return r[b]


    def fit(self, x):
        """
        this function iteratively fits data x into k-means model. The result of the iteration is the cluster centers.
        :param x: input data of (n, m)
        :return: computes self.centers. It also returns sse_veersus_iterations for x.
        """

        # intialize self.centers
        self.init_center(x)

        sse_vs_iter = []
        for iter in range(self.max_iter):
            # finds the cluster index for each x[i] based on the current centers
            labels = self.predict(x)

            # revises the values of self.centers based on the x and current labels
            self.revise_centers(x, labels)

            # computes the sse based on the current labels and centers.
            sse = self.get_sse(x, labels)

            sse_vs_iter.append(sse)

        return sse_vs_iter
