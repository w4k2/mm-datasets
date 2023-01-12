import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

all_genres = np.load("extracted/mmIMDb_genres.npy")
all_genres = all_genres[:, 0]

X_img = np.load("extracted/mmIMDb/mmIMDb_img.npy")
X_txt = np.load("extracted/mmIMDb/mmIMDb_txt.npy")
y = np.load("extracted/mmIMDb/mmIMDb_y.npy")

print("Whole dataset:")
print(tabulate(np.array([all_genres, np.sum(y, axis=0)])))

# Select
# genres = ["Horror", "Romance", "Sci-Fi", "Western", "Drama", "Family"]
genres = ["News"]
genres_idxs = np.array([np.argwhere(all_genres == genre)[0] for genre in genres]).flatten()
# print(genres_idxs)
wuj = np.sum(y[:, genres_idxs], axis=1)
dataset_idxs = np.argwhere(wuj == 1).flatten()

print("\n Selected dataset:")
print(tabulate(np.array([all_genres, np.sum(y[dataset_idxs], axis=0)])))

# Just for fun
# wuj = np.sum(y[:, [17, 5]], axis=1)
# dataset_idxs = np.argwhere(wuj == 2).flatten()
# print(dataset_idxs)

# plt.imshow(X_img[dataset_idxs[0]])
# print(X_txt[dataset_idxs[0]])
# plt.savefig("foo.png")