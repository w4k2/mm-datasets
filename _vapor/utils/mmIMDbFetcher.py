import numpy as np
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from cv2 import resize


class mmIMDbFetcher:
    def __init__(self, data_path, chunk_size=10, n_chunks=1, classes=["Horror", "Romance"], resize=None):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.stream_length = self.chunk_size * self.n_chunks
        self.files = os.listdir(self.data_path)
        self.files = list(set([file.split(".")[0] for file in self.files]))
        self.files.sort()
        self.classes = classes
        self.resize = resize

    def define_classes_stream(self, verbose=True):
        self.class_files = []
        self.labels = []
        for file in tqdm(self.files, disable=not verbose):
            jpeg_path = self.data_path + file + ".jpeg"
            json_path = self.data_path + file + ".json"
            if Path(jpeg_path).exists() and Path(json_path).exists() and (len(self.class_files) < self.stream_length):
                try:
                    json_data = json.load(open(json_path))
                    genres = json_data["genres"]
                    plot = json_data["plot"]
                    # Get intersection
                    inters = list(set(genres).intersection(self.classes))
                    # if any(genre in genres for genre in ):
                    if len(inters) > 0:
                        # self.class_files.append(file)
                        if len(inters) == 1:
                            self.labels.append(self.classes.index(inters[0]))
                            self.class_files.append(file)
                        if len(inters) == 2:
                            print("O CHUJ")
                except:
                    # NO GENRES KEY
                    pass
                
    def define_classes_all(self, verbose=True, size=None):
        self.class_files = []
        self.labels = []
        for file in tqdm(self.files, disable=not verbose):
            jpeg_path = self.data_path + file + ".jpeg"
            json_path = self.data_path + file + ".json"
            if Path(jpeg_path).exists() and Path(json_path).exists():
                try:
                    json_data = json.load(open(json_path))
                    genres = json_data["genres"]
                    plot = json_data["plot"]
                    # Get intersection
                    inters = list(set(genres).intersection(self.classes))
                    # if any(genre in genres for genre in ):
                    if len(inters) > 0:
                        # self.class_files.append(file)
                        if len(inters) == 1:
                            self.labels.append(self.classes.index(inters[0]))
                            self.class_files.append(file)
                        if len(inters) == 2:
                            print("O CHUJ")
                except:
                    # NO GENRES KEY
                    pass

    def load_chunk(self):
        self.chunk_img = []
        self.chunk_txt = []

        if not hasattr(self, "current_sample"):
            self.define_classes_stream()
            print(self.class_files)
            self.current_sample = 0
        else:
            self.current_sample += self.chunk_size

        loaded = 0
        for file in self.class_files[self.current_sample:self.current_sample+self.chunk_size]:
            jpeg_path = self.data_path + file + ".jpeg"
            json_path = self.data_path + file + ".json"
            if Path(jpeg_path).exists() and Path(json_path).exists():
                json_data = json.load(open(json_path))
                txt = ' '.join(str(text) for text in json_data["plot"])
                img = plt.imread(jpeg_path)
                if len(img.shape) == 2:
                    img = np.stack((img, img, img), axis=2)
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                if isinstance(self.resize, tuple):
                    img = resize(img, (self.resize[0], self.resize[1]))
                self.chunk_img.append(img)
                self.chunk_txt.append(txt)

        self.chunk_labels = self.labels[self.current_sample:
                                        self.current_sample+self.chunk_size]

        return (np.array(self.chunk_img), np.array(self.chunk_txt),
                np.array(self.chunk_labels))

    def load_all(self, verbose=True, size=0, mode="all"):
        self.define_classes_all()
        self.all_img = []
        self.all_txt = []
        
        if mode == "all":
            for file in tqdm(self.class_files, disable=not verbose):
                jpeg_path = self.data_path + file + ".jpeg"
                json_path = self.data_path + file + ".json"
                if Path(jpeg_path).exists() and Path(json_path).exists():
                    json_data = json.load(open(json_path))
                    txt = ' '.join(str(text) for text in json_data["plot"])
                    img = plt.imread(jpeg_path)
                    if len(img.shape) == 2:
                        img = np.stack((img, img, img), axis=2)
                    if img.shape[2] == 4:
                        img = img[:, :, :3]
                    if isinstance(self.resize, tuple):
                        img = resize(img, (self.resize[0], self.resize[1]))
                    self.all_img.append(img)
                    self.all_txt.append(txt)
                    
                    if size != 0 and len(self.all_img) == size:
                        return (np.array(self.all_img), np.array(self.all_txt),
                    np.array(self.labels[:size]))
                    
            return (np.array(self.all_img), np.array(self.all_txt),
                    np.array(self.labels))
        if mode == "img":
            for file in tqdm(self.class_files, disable=not verbose):
                jpeg_path = self.data_path + file + ".jpeg"
                json_path = self.data_path + file + ".json"
                if Path(jpeg_path).exists() and Path(json_path).exists():
                    img = plt.imread(jpeg_path)
                    if len(img.shape) == 2:
                        img = np.stack((img, img, img), axis=2)
                    if img.shape[2] == 4:
                        img = img[:, :, :3]
                    if isinstance(self.resize, tuple):
                        img = resize(img, (self.resize[0], self.resize[1]))
                    self.all_img.append(img)
                    
                    if size != 0 and len(self.all_img) == size:
                        return (np.array(self.all_img), np.array(self.labels[:size]))
                    
            return (np.array(self.all_img),
                    np.array(self.labels))
        if mode == "txt":
            for file in tqdm(self.class_files, disable=not verbose):
                jpeg_path = self.data_path + file + ".jpeg"
                json_path = self.data_path + file + ".json"
                if Path(jpeg_path).exists() and Path(json_path).exists():
                    json_data = json.load(open(json_path))
                    txt = ' '.join(str(text) for text in json_data["plot"])
                    self.all_txt.append(txt)
                    
                    if size != 0 and len(self.all_txt) == size:
                        return (np.array(self.all_txt),
                    np.array(self.labels[:size]))
                    
            return (np.array(self.all_txt),
                    np.array(self.labels))
        
        