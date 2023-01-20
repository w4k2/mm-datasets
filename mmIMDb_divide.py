"""
Divide mmIDBd dataset into multiple subsets based on npy format.
"""

import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

all_genres = np.load("data_npy/mmIMDb_genres.npy")
all_genres = all_genres[:, 0]

X_img = np.load("data_npy/mmIMDb/mmIMDb_img.npy")
X_txt = np.load("data_npy/mmIMDb/mmIMDb_txt.npy")
y = np.load("data_npy/mmIMDb/mmIMDb_y.npy")

print("Whole dataset:")
print(tabulate(np.array([all_genres, np.sum(y, axis=0)])))

# Select
datasets = [
    ["Horror", "Romance"],
    ["Crime", "Documentary", "Fantasy", "Sci-Fi"],
    ["Animation", "Biography", "History", "Music", "War"],
    ["Film-Noir", "Musical", "News", "Short", "Sport", "Western"],
    all_genres,
]

for dataset_id, dataset in enumerate(datasets):
    print("\n", dataset)
    if dataset_id == (len(datasets)-1):
        dataset_name = "all"
    else:
        dataset_name = "".join([genre[0] for genre in dataset])
        
    genres_columns = np.array([np.argwhere(all_genres == genre)[0] for genre in dataset]).flatten()
    select = np.sum(y[:, genres_columns], axis=1)
    whole_dataset_idxs = np.argwhere(select == 1).flatten()

    print("\n Selected dataset:")
    print(tabulate(np.array([all_genres, np.sum(y[whole_dataset_idxs], axis=0)])))
    
    dataset_X_img = X_img[whole_dataset_idxs]
    dataset_X_txt = X_txt[whole_dataset_idxs]
    dataset_y = y[whole_dataset_idxs]
    label_encoded_dataset_y = np.argmax(dataset_y[:, genres_columns], axis=1)
    
    print(dataset_X_img.shape)
    print(dataset_X_txt.shape)
    print(label_encoded_dataset_y.shape)
    
    np.save("data_npy/mmIMDb/mmIMDb_%s_img" % dataset_name, np.array(dataset_X_img))
    np.save("data_npy/mmIMDb/mmIMDb_%s_txt" % dataset_name, np.array(dataset_X_txt))
    np.save("data_npy/mmIMDb/mmIMDb_%s_y" % dataset_name, np.array(dataset_y))
    
    # Stacked bar plot
    fig, ax = plt.subplots(1, 1, figsize=(25, 10))
    previous_scores = 0
    for genre_id, one_genre in enumerate(dataset):
        one_genre_idxs = np.argwhere(dataset_y[:, genres_columns[genre_id]] == 1).flatten()
        if genre_id == 0:
            ax.bar(all_genres, np.sum(dataset_y[one_genre_idxs], axis=0), 0.35, label=one_genre)
        else:
            ax.bar(all_genres, np.sum(dataset_y[one_genre_idxs], axis=0), 0.35, label=one_genre, bottom=previous_scores)
        previous_scores += np.sum(dataset_y[one_genre_idxs], axis=0)
    ax.set_ylim(0, round(np.max(np.sum(dataset_y, axis=0)), -3)+500)
    ax.set_ylabel("# Films")
    plt.grid(ls=":", c=(.7, .7, .7))
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("figures/mmIMDb_%s.png" % dataset_name, dpi=200)