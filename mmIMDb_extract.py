from utils import mmIMDbFetcher
from os import listdir
import numpy as np
import os
import json
from tqdm import tqdm


data_path = "/Volumes/T7/Multimodal/mmIMDb/"

# Get all genres with count
if os.path.isfile("utils/mmIMDb_genres.npy"):
    mmIMDb_genres = np.load("utils/mmIMDb_labels.npy")
else:
    names = os.listdir(data_path)
    names = np.unique(np.array([name[:-5] for name in names]))[1:]
    print(len(names))
    total = []
    for name in tqdm(names):
        try:
            # print(name)
            json_data = json.load(open("%s/%s.json" % (data_path, name)))
            total.append(json_data['genres'])
        except:
            print("NOT AVAILABLE!")

    genres, counts = np.unique(sum(total, []), return_counts=True)

    mmIMDb_genres = np.concatenate((genres.reshape(-1, 1), counts.reshape(-1, 1)), axis=1)
    np.save("utils/mmIMDb_genres", mmIMDb_genres)

print(mmIMDb_genres)
exit()
# Load data
data_path = "/Volumes/T7/Multimodal/mmIMDb/"
size = 0
batch_size = 2000
# HR
classes = ["Horror", "Romance"]
# classes = ["Fantasy", "Western"]
# classes = ["Musical", "Sport"]
data = mmIMDbFetcher(data_path, classes=classes, resize=(224, 224))

X_img, y = data.load_all(size=size, mode="img")