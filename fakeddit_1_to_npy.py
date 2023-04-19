"""

"""
import tarfile
from PIL import Image
import io
import numpy as np
import pandas as pd
from cv2 import resize
from torchvision.models import ResNet18_Weights
from torch import from_numpy
import matplotlib.pyplot as plt


root = "../fakeddit/"
# TXT & y
txt_files = ["multimodal_test_public.tsv", "multimodal_validate.tsv", "multimodal_train.tsv"]

txts = []
ys = []
for file in txt_files:
    data = pd.read_csv(root + file, delimiter='\t', on_bad_lines='skip', index_col=False, nrows=10)
    txt = data["clean_title"].to_numpy()
    print(txt[:10])
    y_2 = data["2_way_label"]
    y_3 = data["3_way_label"]
    y_6 = data["6_way_label"]
    y = np.vstack((y_2, y_3, y_6)).T
    txts.append(txt)
    ys.append(y)
txts = np.concatenate(txts)
ys = np.concatenate(ys)

# Comments
data = pd.read_csv(root + "all_comments.tsv", delimiter='\t', on_bad_lines='skip', index_col=False, nrows=10)
comms = data["body"].to_numpy()
print(comms[:10])
# IMG
tar = tarfile.open(root + "public_images.tar.bz2")
imgs = []
for id, member in enumerate(tar):
    # Folderu nie chcemy
    if id > 0:
        print(member.name)
        image=tar.extractfile(member)
        image = image.read()
        image = np.asarray(Image.open(io.BytesIO(image)))
        if len(image.shape) == 2:
            image = np.stack((image, image, image), axis=2)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        print(image.shape)
        resnet18_weights = ResNet18_Weights.IMAGENET1K_V1
        resnet18_transforms = resnet18_weights.transforms()
        image = np.swapaxes(resnet18_transforms(from_numpy(np.swapaxes(np.array(image), 0, 2))).numpy(), 0, 2)
        print(image.shape)
        imgs.append(image)
        plt.imshow(image)
        plt.savefig("foo.png")
        if id == 11:
            exit()

imgs = np.array(imgs)

np.save("data_npy/fakeddit/fakeddit_img", np.array(imgs))
np.save("data_npy/fakeddit/fakeddit_txt", np.array(txts))
np.save("data_npy/fakeddit/fakeddit_comm", np.array(comms))
np.save("data_npy/fakeddit/fakeddit_y", np.array(ys))