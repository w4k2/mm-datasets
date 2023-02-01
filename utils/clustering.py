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
                    
                    
                    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    # ax.scatter(X[:,0], X[:,1])
                    # ax.scatter(self.centroids_[:,0], self.centroids_[:,1], c='red')
                    # plt.tight_layout()
                    # plt.savefig("foo.png")
                    # plt.close()
                    # sleep(.1)
                    # exit()
                if self.init == 'k-means++':
                    # first centroid
                    self.centroids_ = self.X_[self.random.randint(0, X.shape[0], 1)]
                    while len(self.centroids_) < self.n_clusters:
                        # distance from each point to nearest centroid
                        self.nearest_distance_ = np.min(cdist(from_numpy(self.X_), from_numpy(self.centroids_), p=self.p).numpy(), axis=1)
                        # probability normalization
                        self.nearest_distance_ = np.nan_to_num(self.nearest_distance_, nan=0.0000000000001)
                        self.init_proba_ = self.nearest_distance_/np.sum(self.nearest_distance_)
                        next_centroid = self.random.choice(np.arange(0, self.X_.shape[0], 1), 1, p=self.init_proba_)
                        self.centroids_ = np.append(self.centroids_, self.X_[next_centroid], axis=0)
                    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    # ax.scatter(X[:,0], X[:,1])
                    # ax.scatter(self.centroids_[:,0], self.centroids_[:,1], c='red', s=50)
                    # plt.tight_layout()
                    # plt.savefig("foo.png")
                    # plt.close()
                    # sleep(.1)
                    # exit()
            
            # print("RAZ")
            # print(self.centroids_.shape)
            
            self.distance_matrix_ = cdist(from_numpy(self.X_), from_numpy(self.centroids_), p=self.p).numpy()
            # print("DWA")
            # print(self.distance_matrix_.shape)
            self.cluster_affiliation_ = np.argmin(self.distance_matrix_, axis=1)
            # print("TRZY")
            # print(np.unique(self.cluster_affiliation_, return_counts=True))
            prev_centroids = self.centroids_
            # print("CZTERY")
            # print(prev_centroids, prev_centroids.shape)
            # put a previous centroid in an empty cluster
            self.centroids_ = []
            for c in range(self.n_clusters):
                if self.X_[self.cluster_affiliation_ == c].shape[0] > 0:
                    self.centroids_.append(np.mean(self.X_[self.cluster_affiliation_ == c], axis=0))
                else:
                    self.centroids_.append(prev_centroids[c])
            self.centroids_ = np.array(self.centroids_)
            # self.centroids_ = np.array([np.mean(self.X_[self.cluster_affiliation_ == c], axis=0) for c in range(self.n_clusters)])
            
            # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            # ax.scatter(X[:,0], X[:,1], c=self.cluster_affiliation_)
            # ax.scatter(self.centroids_[:,0], self.centroids_[:,1], c='red', s=50)
            # plt.tight_layout()
            # plt.savefig("foo.png")
            # plt.close()
            # sleep(.1)
            
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