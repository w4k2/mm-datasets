import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torch import from_numpy
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import IndicesDataset, kMeans, purity_score, LinearClassifier
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, StandardScaler
from torch import cdist


root = "../../mm-datasets/data_npy/"
dataset_names = ["sport", "instruments", "riding", "dancing", "eating"]
modalities = ["audio", "video"]

for dataset_id, dataset in enumerate(dataset_names):
    print(dataset)
    y = np.load(root + "kinetics400/kinetics400_y_%s.npy" % dataset)
    ids = np.load(root + "kinetics400/kinetics400_ids_%s.npy" % dataset)
    for modality in modalities:
        print(modality)
        X = np.load(root + "kinetics400/kinetics400_%s_%s.npy" % (modality, dataset))
        
    
        X_train, X_extract, y_train, y_extract, ids_train, ids_extract = train_test_split(
            X, y, ids, test_size=.8, random_state=1410, stratify=y, shuffle=True)
        print("Ile?")
        print(X_train.shape)
        """
        Training
        """
        num_classes = np.unique(y).shape[0]
        batch_size = 8
        num_epochs = 100
        weights = ResNet18_Weights.IMAGENET1K_V1
        # weights = None
        n_clusters = 50
        n_components = 256

        model = resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        device = torch.device("mps")
        model = model.to(device)

        params_to_update = model.parameters()
        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        _, counts = np.unique(y_train, return_counts=True)
        criterion = nn.CrossEntropyLoss()

        dataset_ready = IndicesDataset(TensorDataset(from_numpy(np.moveaxis(X_train, 3, 1)).float(), from_numpy(y_train).long()))
        data_loader = DataLoader(dataset_ready, batch_size=batch_size, shuffle=True)

        model.train()
        print("TRAINING!")
        all_loss = np.zeros(num_epochs)
        all_nmi = np.zeros(num_epochs)
        kmeans = KMeans(n_clusters=n_clusters)
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0            
            
            """
            Deep cluster
            """
            return_nodes = {
                'flatten': 'extracted_flatten',
            }
            model.eval()
            extractor = create_feature_extractor(model, return_nodes=return_nodes)
            print("Epoch extraction!")
            X_train_extracted = extractor(from_numpy(np.moveaxis(X_train, 3, 1)).float().to(device))["extracted_flatten"].cpu().detach().numpy()
            
            pca = PCA(n_components, random_state=1410)
            X_train_extracted = pca.fit_transform(X_train_extracted)
            normalizer = Normalizer()
            X_train_extracted = normalizer.fit_transform(X_train_extracted)
            
            cluster_labels = kmeans.fit_predict(X_train_extracted)
            
            print(y_train)
            print(cluster_labels)
            # print(kmeans.cluster_centers_)
            
            all_nmi[epoch] = normalized_mutual_info_score(y_train, cluster_labels)
            
            model.train()
            model.fc = nn.Linear(num_ftrs, num_classes).to(device)
            for i, batch in enumerate(data_loader, 0):
                inputs, _, idxs = batch
                labels = from_numpy(np.array(cluster_labels[idxs]).reshape(1, -1).flatten())
                
                optimizer.zero_grad()

                outputs = model(inputs.to(device))
                loss = criterion(outputs.to(device), labels.to(device))
                epoch_loss += loss.cpu().detach().numpy()
                loss.backward()
                
                optimizer.step()
                
            all_loss[epoch] = epoch_loss / (i+1)
            fig, ax = plt.subplots(2, 2, figsize = (20, 20))
            ax[0,0].plot([i for i in range(1, num_epochs+1)][:epoch+1], all_loss[:epoch+1], c = "dodgerblue")
            ax[0,1].plot([i for i in range(1, num_epochs+1)][:epoch+1], all_nmi[:epoch+1], c = "red")
            ax[0,0].set_xlim(1, num_epochs)
            ax[0,1].set_xlim(1, num_epochs)
            
            pca = PCA(2, random_state=1410)
            plot = pca.fit_transform(X_train_extracted)
            centres = pca.transform(kmeans.cluster_centers_)
            ax[1,0].scatter(plot[:, 0], plot[:, 1], c=cluster_labels)
            ax[1,0].scatter(centres[:, 0], centres[:, 1], c="red")
            
            ax[1,1].scatter(plot[:, 0], plot[:, 1], c=y_train)
            ax[1,1].scatter(centres[:, 0], centres[:, 1], c="red")
            plt.tight_layout()
            plt.savefig("foo.png")
            plt.close()
        """
        Extraction
        """
        X_extract = from_numpy(np.moveaxis(X_extract, 3, 1)).float()
        model.eval()

        all_extracted = []
        current_sample = 0
        batch_size = 500
        print("EXTRACION!")
        while current_sample < X_extract.shape[0]:
            print("Batch %i:%i" % (current_sample, current_sample+batch_size))
            X_extract_batch = X_extract[current_sample:current_sample+batch_size]
            return_nodes = {
                'flatten': 'extracted_flatten',
            }
            extractor = create_feature_extractor(model, return_nodes=return_nodes)
            X_batch_extracted = extractor(X_extract_batch.to(device))["extracted_flatten"].cpu().detach().numpy()
            
            
            all_extracted.append(X_batch_extracted)
            current_sample += X_extract_batch.shape[0]
            
        all_extracted = np.vstack(tuple(all_extracted))
        
    #     np.save("data_extracted/kinetics400/kinetics400_%s_%s" % (modality, dataset_names[dataset_id]), all_extracted)
    # np.save("data_extracted/kinetics400/kinetics400_y_%s" % dataset_names[dataset_id], y_extract)
    # np.save("data_extracted/kinetics400/kinetics400_ids_%s" % dataset_names[dataset_id], ids_extract)
        
    print("\n")