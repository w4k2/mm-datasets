from utils import mmIMDbFetcher
from os import listdir

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