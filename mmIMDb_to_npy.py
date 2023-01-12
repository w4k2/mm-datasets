import os
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
from cv2 import resize

data_path = "/Volumes/T7/Multimodal/mmIMDb/"
files = os.listdir(data_path)
files = list(set([file.split(".")[0] for file in files]))
files.sort()

all_genres = np.load("extracted/mmIMDb_genres.npy")
print(all_genres)

mmIMDb_img = []
mmIMDb_txt = []
mmIMDb_y = []

for file in tqdm(files):
    jpeg_path = data_path + file + ".jpeg"
    json_path = data_path + file + ".json"
    if Path(jpeg_path).exists() and Path(json_path).exists():
        try:
            json_data = json.load(open(json_path))
            
            # LOAD KEYS
            movie_genres = json_data["genres"]
            txt = ' '.join(str(text) for text in json_data["plot"])
            
            # ENCODED y
            # encoded_movie_genres = [1 if movie_genre in movie_genres else 0 for movie_genre in all_genres[:, 0]]
            # mmIMDb_y.append(encoded_movie_genres)
            
            # IMG
            img = plt.imread(jpeg_path)
            if len(img.shape) == 2:
                img = np.stack((img, img, img), axis=2)
            if img.shape[2] == 4:
                img = img[:, :, :3]
            img = resize(img, (224, 224))
            mmIMDb_img.append(img)
            
            # TXT
            # mmIMDb_txt.append(txt)
            
        except:
            # NO KEYS
            print("NO KEYS!")
            pass
        
np.save("extracted/mmIMDb/mmIMDb_img", np.array(mmIMDb_img))
# np.save("extracted/mmIMDb/mmIMDb_txt", np.array(mmIMDb_txt))
# np.save("extracted/mmIMDb/mmIMDb_y", np.array(mmIMDb_y))