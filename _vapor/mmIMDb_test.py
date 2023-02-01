import numpy as np
import matplotlib.pyplot as plt

all_genres = np.load("extracted/mmIMDb_genres.npy")
all_genres = all_genres[:, 0]
print(all_genres)

X_img = np.load("extracted/mmIMDb/mmIMDb_img.npy")
X_txt = np.load("extracted/mmIMDb/mmIMDb_txt.npy")
y = np.load("extracted/mmIMDb/mmIMDb_y.npy")

plt.imshow(X_img[26000])
plt.savefig("foo.png")
print(X_txt[26000])
print(y[26000])
print(all_genres[np.argwhere(y[26000]==1)])

print(np.sum(y, axis=0))