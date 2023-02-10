import numpy as np
from torch import cdist, from_numpy
from sklearn.base import BaseEstimator, ClusterMixin


class kMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=2, p=2, max_iter=300, tol=1e-4, init="random", random_state=None):
        self.n_clusters = n_clusters
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.random = np.random.RandomState(seed=self.random_state)
        
    def fit(self, X):
        self.X_ = X
        
        for iter in range(self.max_iter):
            if not hasattr(self, "centroids_"):
                if self.init == 'random':
                    self.centroids_ = self.X_[self.random.randint(0, X.shape[0], self.n_clusters)]
            
            self.distance_matrix_ = cdist(from_numpy(self.X_), from_numpy(self.centroids_), p=self.p).numpy()
            self.cluster_affiliation_ = np.argmin(self.distance_matrix_, axis=1)
            prev_centroids = self.centroids_
            self.centroids_ = []
            for c in range(self.n_clusters):
                if self.X_[self.cluster_affiliation_ == c].shape[0] > 0:
                    self.centroids_.append(np.mean(self.X_[self.cluster_affiliation_ == c], axis=0))
                else:
                    self.centroids_.append(prev_centroids[c])
            self.centroids_ = np.array(self.centroids_)
            
            if np.linalg.norm(self.centroids_ - prev_centroids) < self.tol:
                break
            
        return self
    
    def predict(self, X):
        self.distance_matrix_ = cdist(from_numpy(X), from_numpy(self.centroids_), p=self.p).numpy()
        pred = np.argmin(self.distance_matrix_, axis=1)
        return pred
    
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)