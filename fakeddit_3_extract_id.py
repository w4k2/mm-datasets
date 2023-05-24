import tarfile
from PIL import Image
import PIL
import io
import numpy as np
import pandas as pd
from cv2 import resize
from torchvision.models import ResNet18_Weights
from torch import from_numpy
import matplotlib.pyplot as plt
import traceback
from tqdm import tqdm
import sys

root = "../fakeddit/"

post_sample = pd.read_csv('data_csv/fakeddit/posts.csv').astype({'id': str})

tar = tarfile.open(root + "public_images.tar.bz2")
imgs = []
labels = []

total = 0
errors = []
idx = 0

for id, member in tqdm(enumerate(tar)):
    # Folderu nie chcemy
    if id > 0:
        name = member.name.split('/')[1][:-4]
        if name in post_sample['id'].values:
            total += 1
            try:
                image=tar.extractfile(member)
                image = image.read()
                image = np.asarray(Image.open(io.BytesIO(image)))
                if len(image.shape) == 2:
                    image = np.stack((image, image, image), axis=2)
                if image.shape[2] == 4:
                    image = image[:, :, :3]
                # print(image.shape)
                resnet18_weights = ResNet18_Weights.IMAGENET1K_V1
                resnet18_transforms = resnet18_weights.transforms()
                image = np.swapaxes(resnet18_transforms(from_numpy(np.swapaxes(np.array(image), 0, 2))).numpy(), 0, 2)

                labels.append(post_sample.loc[post_sample['id'] == name,['2_way_label','3_way_label','6_way_label', 'id']].values.ravel())

            except PIL.UnidentifiedImageError:
                errors.append(member.name)
                print(f'Error {member.name}')
                traceback.print_exc()

np.save(f"data_npy/fakeddit/fakeddit_img_y_id", np.array(labels))
