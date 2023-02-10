import numpy as np

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from sklearn.model_selection import train_test_split
from torch import from_numpy
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from utils import kMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from itertools import permutations
from scipy.stats import mode


class IndicesDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        data, target = self.dataset[index]
        
        return data, target, index

    def __len__(self):
        return len(self.dataset)


# X_video = np.load("data_npy/kinetics400/kinetics400_audio_sport.npy")
X_audio = np.load("data_npy/kinetics400/kinetics400_audio_sport.npy")
X_video = X_audio
y = np.load("data_npy/kinetics400/kinetics400_y_sport.npy")
print(X_video.shape)
print(X_audio.shape)
print(y.shape)

# print(np.unique(y))
# perms = [np.array(i) for i in permutations(np.unique(y)[:3])]
# print(perms)
# exit()

X_train, X_test, y_train, y_test = train_test_split(
    X_video, y, test_size=.8, random_state=1410, stratify=y, shuffle=True)

# Model
num_classes = np.unique(y).shape[0]
batch_size = 8
num_epochs = 1
weights = ResNet18_Weights.IMAGENET1K_V1
# weights = None

model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device("mps")
model = model.to(device)

params_to_update = model.parameters()
optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

dataset = IndicesDataset(TensorDataset(from_numpy(np.moveaxis(X_train, 3, 1)).float(), from_numpy(y_train).long()))
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model.train()
print("TRAINING!")
all_loss = np.zeros(num_epochs)
kmeans = kMeans(n_clusters=num_classes, p=2)
for epoch in tqdm(range(num_epochs)):
    epoch_loss = 0
    
    # DeepCluster
    return_nodes = {
        'flatten': 'extracted_flatten',
    }
    model.eval()
    extractor = create_feature_extractor(model, return_nodes=return_nodes)
    print("Epoch extraction!")
    X_train_extracted = extractor(from_numpy(np.moveaxis(X_train, 3, 1)).float().to(device))["extracted_flatten"].cpu().detach().numpy()
    
    pca = PCA(2)
    X_train_extracted_pca = pca.fit_transform(X_train_extracted)
    # normalizer = Normalizer(norm="l2")
    # X_train_extracted = normalizer.fit_transform(X_train_extracted)
    # print("Epoch clustering!")
    
    cluster_labels = kmeans.fit_predict(X_train_extracted)
    model.train()
    for i, batch in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels, indices]
        inputs, _, idxs = batch
        labels = from_numpy(np.array(cluster_labels[idxs]).reshape(1, -1).flatten())
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.to(device))
        loss = criterion(outputs.to(device), labels.to(device))
        epoch_loss += loss.cpu().detach().numpy()
        loss.backward()
        optimizer.step()
        
    all_loss[epoch] = epoch_loss / (i+1)
    fig, ax = plt.subplots(2, 1, figsize = (20, 20))
    ax[0].plot([i for i in range(1, num_epochs+1)][:epoch+1], all_loss[:epoch+1], c = "dodgerblue")
    ax[0].set_xlim(1, num_epochs)
    ax[1].scatter(X_train_extracted_pca[:, 0], X_train_extracted_pca[:, 1], c=cluster_labels)
    # ax[1].scatter(pca.transform(kmeans.centroids_)[:, 0], pca.transform(kmeans.centroids_)[:, 1], c="red", lw = 5)
    plt.tight_layout()
    plt.savefig("foo.png")
    plt.close()
    # exit()

X_test = from_numpy(np.moveaxis(X_test, 3, 1)).float()
model.eval()

all_preds = []
current_sample = 0
batch_size = 500
print("PREDICTION!")
while current_sample < X_test.shape[0]:
    print("Batch %i:%i" % (current_sample, current_sample+batch_size))
    X_test_batch = X_test[current_sample:current_sample+batch_size]
    logits = model(X_test_batch.to(device))
    probs = nn.functional.softmax(logits.cpu(), dim=1).detach().numpy()
    # logits = model(X_test_batch)
    # probs = nn.functional.softmax(logits, dim=1).detach().numpy()
    preds = np.argmax(probs, 1)
    all_preds.append(preds)
    current_sample += X_test_batch.shape[0]
    # print(X_test_batch.shape)
    # score =  balanced_accuracy_score(y_test[current_sample:current_sample+batch_size], preds)
    # print(score)
    
all_preds = [item for sublist in all_preds for item in sublist]

encoded_classes = []
for cluster in range(num_classes):
    cluster_samples = np.argwhere(cluster_labels == cluster)
    new_class = mode(y_test[cluster_samples])
    encoded_classes.append(new_class)
    print(cluster)
    print(new_class)
exit()
all_preds = np.array(all_preds)
# print(all_preds)
score =  balanced_accuracy_score(y_test, all_preds)
print(score)

# model.eval()

# return_nodes = {
#     'flatten': 'extracted_flatten',
# }
# extractor = create_feature_extractor(model, return_nodes=return_nodes)
# wuj = from_numpy(np.moveaxis(X_train[:10], 3, 1)).float()
# X_extracted = extractor(wuj)["extracted_flatten"].detach().numpy()
# print(X_extracted.shape)