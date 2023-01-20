"""
Estimating optimal cluster number for the mmIMDb datasets using  Calinski Harabasz Score.
"""
from sklearn.metrics import calinski_harabasz_score
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# load data
all_genres = np.load("data_npy/mmIMDb_genres.npy")
all_genres = all_genres[:, 0]

datasets = [
    ["Horror", "Romance"],
    ["Crime", "Documentary", "Fantasy", "Sci-Fi"],
    ["Animation", "Biography", "History", "Music", "War"],
    ["Film-Noir", "Musical", "News", "Short", "Sport", "Western"],
    all_genres,
]
modalities = ["txt", "img"]
pca = PCA(n_components=.7)

# Define all k
all_k_clusters = np.arange(2, 30, 2)

fig, ax = plt.subplots(len(datasets), 1, figsize=(20, 60))
for dataset_id, dataset in tqdm(enumerate(datasets)):
    
    if dataset_id == (len(datasets)-1):
        dataset_name = "all"
    else:
        dataset_name = "".join([genre[0] for genre in dataset])
        
    for modality in modalities:
        X = np.load("data_extracted/mmIMDb/mmIMDb_%s_%s.npy" % (dataset_name, modality))
        X = pca.fit_transform(X)
        chs_scores = []
        for k in tqdm(all_k_clusters):
            kmeans = KMeans(n_clusters=k).fit(X)
            chs_scores.append(calinski_harabasz_score(X, kmeans.labels_))
        ax[dataset_id].plot(all_k_clusters, chs_scores, label=modality)
        ax[dataset_id].set_title(dataset_name)
        ax[dataset_id].set_ylabel("Calinski Harabasz Score", fontsize = 15)
        ax[dataset_id].set_xlabel("n clusters", fontsize = 15)
plt.tight_layout()
plt.legend(frameon=False, fontsize=20)
plt.savefig("figures/chs.png", dpi=200)
plt.close()
    