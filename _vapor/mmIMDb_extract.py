from utils import mmIMDbFetcher

from os import listdir
import numpy as np
import os
import json
from tqdm import tqdm
from itertools import combinations


data_path = "/Volumes/T7/Multimodal/mmIMDb/"

# Get all genres with count
if os.path.isfile("extracted/mmIMDb_genres.npy"):
    mmIMDb_genres = np.load("extracted/mmIMDb_genres.npy")
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
    np.save("extracted/mmIMDb_genres", mmIMDb_genres)

print(mmIMDb_genres)

# Dataset choice
"""
n_classes_for_datasets = np.linspace(2, mmIMDb_genres.shape[0], 5).astype(int)
random = np.random.RandomState()
for n_classes in n_classes_for_datasets:
    comb = np.array([list(i) for i in combinations(mmIMDb_genres[:, 0], n_classes)])
    chosen = random.randint(0, len(comb), 1)
    print(comb[chosen])
# exit()
"""

# Load data
data_path = "/Volumes/T7/Multimodal/mmIMDb/"
size = 0

datasets = [["Horror", "Romance"],
            
            ['Animation', 'Documentary', 'Fantasy', 'History', 'Horror', 'Short', 'Sport',
            'Thriller'],
            
            ['Animation', 'Biography', 'Comedy', 'Family', 'History', 'Music', 'Mystery',
            'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'Western'],
            
            ['Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary',
            'Drama', 'Fantasy', 'History', 'Music', 'News', 'Reality-TV', 'Romance',
            'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western'],
            
            ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror',
            'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi',
            'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']]

for dataset in datasets:
    print(dataset)
    data = mmIMDbFetcher(data_path, classes=dataset, resize=(224, 224))
    X_img, X_txt, y = data.load_all(size=size, mode="all")
    
    np.save("extracted/mmIMDb/mmIMDb_%i_img" % (len(dataset)), X_img)
    np.save("extracted/mmIMDb/mmIMDb_%i_txt" % (len(dataset)), X_txt)
    np.save("extracted/mmIMDb/mmIMDb_%i_y" % (len(dataset)), y)
