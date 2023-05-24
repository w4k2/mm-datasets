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

# TEXT ------------------------

# # Load posts
# txt_files = ["multimodal_test_public.tsv", "multimodal_validate.tsv", "multimodal_train.tsv"]

# post_data = None
# for file in txt_files:
#     tmp_data = pd.read_csv(root + file, delimiter='\t', on_bad_lines='skip', index_col=False)

#     if post_data is None:
#         post_data = tmp_data
#     else:
#         post_data = pd.concat([post_data, tmp_data], ignore_index=True)

# # Load comments
# comment_data = pd.read_csv(root + "all_comments.tsv", delimiter='\t', on_bad_lines='skip', index_col=False, low_memory=True)

# # Sample data
# comment_data = comment_data[comment_data['isTopLevel'] == True]
# data_com = comment_data['submission_id'].value_counts()[5000:60000]
# select = post_data['id'].isin(data_com.index.values)
# post_sample = post_data[select.values]
# select = comment_data['submission_id'].isin(data_com.index.values)
# comment_sample = comment_data[select.values].drop(columns=['Unnamed: 0'])

# # Plot posts/comments
# ax = data_com.hist(bins=25)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.xlabel('Comments per post')
# plt.ylabel('Number of posts')
# plt.grid(color='gray', linestyle='--', linewidth=0.5)
# plt.savefig('foo.png')

# # Post label
# y_2 = post_sample["2_way_label"]
# y_3 = post_sample["3_way_label"]
# y_6 = post_sample["6_way_label"]
# y_post = np.vstack((y_2, y_3, y_6)).T

# # Comment label
# comment_sample = comment_sample.merge(post_sample[['id','2_way_label','3_way_label','6_way_label']], suffixes=('', '_post'), left_on='submission_id', right_on='id', how='right')
# y_2 = comment_sample["2_way_label"]
# y_3 = comment_sample["3_way_label"]
# y_6 = comment_sample["6_way_label"]
# y_comment = np.vstack((y_2, y_3, y_6)).T

# # Save to csv
# post_sample.to_csv('data_csv/fakeddit/posts.csv', index=0)
# comment_sample.to_csv('data_csv/fakeddit/comments.csv', index=0)

# # Save to numpy
# np.save(file='data_npy/fakeddit/fakeddit_posts', arr=post_sample['clean_title'].values)
# np.save(file='data_npy/fakeddit/fakeddit_posts_y', arr=y_post)
# np.save(file='data_npy/fakeddit/fakeddit_comments', arr=comment_sample['body'].values)
# np.save(file='data_npy/fakeddit/fakeddit_comments_y', arr=y_comment)

# IMAGE ------------------------

# Load csv
post_sample = pd.read_csv('data_csv/fakeddit/posts.csv').astype({'id': str})

# Load images
tar = tarfile.open(root + "public_images.tar.bz2")
imgs = []
labels = []

total = 0
errors = []
idx = 0
# print(tar.getnames())
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

                resnet18_weights = ResNet18_Weights.IMAGENET1K_V1
                resnet18_transforms = resnet18_weights.transforms()
                image = np.swapaxes(resnet18_transforms(from_numpy(np.swapaxes(np.array(image), 0, 2))).numpy(), 0, 2)
                imgs.append(image)
                
                labels.append(post_sample.loc[post_sample['id'] == name,['2_way_label','3_way_label','6_way_label', 'id']].values.ravel())
            
            except PIL.UnidentifiedImageError:
                errors.append(member.name)
                print(f'Error {member.name}')
                traceback.print_exc()
        
        if id % 100 == 0:
            print(f' Size: {sys.getsizeof(imgs)} bytes for {len(imgs)} elements -- {total} total.')
            if sys.getsizeof(imgs) > 60000:
                imgs = np.array(imgs)
                # print(imgs.shape, np.array(labels).shape)
                # np.save(f"data_npy/fakeddit/fakeddit_img_{idx}_{imgs.shape[0]}", np.array(imgs))
                # np.save(f"data_npy/fakeddit/fakeddit_img_y_{idx}", np.array(labels))
                np.savez_compressed(f"data_npy/fakeddit/fakeddit_img_{idx}", X=np.array(imgs), y=np.array(labels))
                idx +=1
                imgs = []
                labels = []

imgs = np.array(imgs)
np.savez_compressed(f"data_npy/fakeddit/fakeddit_img_{idx}", X=np.array(imgs), y=np.array(labels))


print(f'Total {total} -- error in {len(errors)}')
errors = np.array(errors)
np.savetxt('data_csv/fakeddit/errors.csv', errors)