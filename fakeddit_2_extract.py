"""
Using deep extractors to extract features from each predefined dataset.
Do zrobienia
"""
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from sentence_transformers import SentenceTransformer
from torch import from_numpy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Batch size for extraction processing
batch_size = 2000

X_post = np.load('data_npy/fakeddit/fakeddit_posts.npy', allow_pickle=True)
y_post = np.load('data_npy/fakeddit/fakeddit_posts_y.npy', allow_pickle=True).astype(int)
X_comment = np.load('data_npy/fakeddit/fakeddit_comments.npy', allow_pickle=True)
y_comment = np.load('data_npy/fakeddit/fakeddit_comments_y.npy', allow_pickle=True).astype(int)


# """
# Text to embeddings
# """

# # Posts
# print("Extracting TXT from posts!")
# print(X_post.shape)
# embedder = SentenceTransformer('all-MiniLM-L6-v2')
# current_sample = 0
# batch_embeddings = []
# print("EXTRACION!")
# while current_sample < X_post.shape[0]:
#     print("Batch %i:%i" % (current_sample, current_sample+batch_size))
#     X_post_batch = X_post[current_sample:current_sample+batch_size]
#     embeddings = embedder.encode(X_post_batch)
#     batch_embeddings.append(embeddings)
#     current_sample += X_post_batch.shape[0]
#     # print("CURRENT SAMPLE: %i" % current_sample)

# corpus_embeddings = np.concatenate(batch_embeddings, axis=0)
# print("TXT from posts extracted!")
# print(corpus_embeddings.shape)
# np.save("data_extracted/fakeddit/fakeddit_posts", corpus_embeddings)

# print("y from posts extracted!")
# print(y_post.shape)
# np.save("data_extracted/fakeddit/fakeddit_posts_y", y_post)


# # Comments
# print("Extracting TXT from comments!")
# print(X_comment.shape)
# embedder = SentenceTransformer('all-MiniLM-L6-v2')
# current_sample = 0
# batch_embeddings = []
# print("EXTRACION!")
# while current_sample < X_comment.shape[0]:
#     print("Batch %i:%i" % (current_sample, current_sample+batch_size))
#     X_comment_batch = X_comment[current_sample:current_sample+batch_size]
#     embeddings = embedder.encode(X_comment_batch)
#     batch_embeddings.append(embeddings)
#     current_sample += X_comment_batch.shape[0]
#     # print("CURRENT SAMPLE: %i" % current_sample)

# corpus_embeddings = np.concatenate(batch_embeddings, axis=0)
# print("TXT from comments extracted!")
# print(corpus_embeddings.shape)
# np.save("data_extracted/fakeddit/fakeddit_comments", corpus_embeddings)

# print("y from comments extracted!")
# print(y_comment.shape)
# np.save("data_extracted/fakeddit/fakeddit_comments_y", y_comment)


"""
Extraction from images
"""

weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)

device = torch.device("mps")
model = model.to(device)
model.eval()
# Extract
for idx in range(5):
    img_npz = np.load(f'data_npy/fakeddit/fakeddit_img_{idx}.npz', allow_pickle=True)

    X = from_numpy(np.moveaxis(img_npz['X'], 3, 1)).float()
    y = img_npz['y']

    all_extracted = []
    current_sample = 0
    batch_size = 500
    print("EXTRACION!")
    while current_sample < X.shape[0]:
        print("Batch %i:%i" % (current_sample, current_sample+batch_size))
        X_img_extract_batch = X[current_sample:current_sample+batch_size]
        return_nodes = {
            'flatten': 'extracted_flatten',
        }
        extractor = create_feature_extractor(model, return_nodes=return_nodes)
        X_img_batch_extracted = extractor(X_img_extract_batch.to(device))["extracted_flatten"].cpu().detach().numpy()
        
        
        all_extracted.append(X_img_batch_extracted)
        current_sample += X_img_extract_batch.shape[0]
        
    all_extracted = np.vstack(tuple(all_extracted))
    print("IMG extracted!")
    print(all_extracted.shape)
    np.savez_compressed(f'data_extracted/fakeddit/img_{idx}', X=all_extracted, y=y)

    
