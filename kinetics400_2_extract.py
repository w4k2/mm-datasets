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

dataset_names = ["sport", "instruments", "riding", "dancing", "eating"]
modalities = ["video", "audio"]

for dataset_id, dataset in enumerate(dataset_names):
    print(dataset)
    y = np.load("data_npy/kinetics400/kinetics400_y_%s.npy" % dataset)
    ids = np.load("data_npy/kinetics400/kinetics400_ids_%s.npy" % dataset)
    for modality in modalities:
        print(modality)
        X = np.load("data_npy/kinetics400/kinetics400_%s_%s.npy" % (modality, dataset))
        
    
        X_train, X_extract, y_train, y_extract, ids_train, ids_extract = train_test_split(
            X, y, ids, test_size=.8, random_state=1410, stratify=y, shuffle=True)
        print(y_extract.shape)
        print(y_extract[:20])
        # Train model
        num_classes = np.unique(y).shape[0]
        batch_size = 8
        num_epochs = 10
        weights = ResNet18_Weights.IMAGENET1K_V1

        model = resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        device = torch.device("mps")
        model = model.to(device)

        params_to_update = model.parameters()
        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        _, counts = np.unique(y_train, return_counts=True)
        weights = from_numpy(np.array([1-(count/np.sum(counts)) for count in counts])).float().to(device)
        print(weights)
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.CrossEntropyLoss(weight = weights)

        dataset_ready = TensorDataset(from_numpy(np.moveaxis(X_train, 3, 1)).float(), from_numpy(y_train).long())
        data_loader = DataLoader(dataset_ready, batch_size=batch_size, shuffle=True)

        model.train()
        print("TRAINING!")
        all_loss = np.zeros(num_epochs)
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0            
            for i, batch in enumerate(data_loader, 0):
                # get the inputs; data is a list of [inputs, labels, indices]
                inputs, labels = batch
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs.to(device))
                loss = criterion(outputs.to(device), labels.to(device))
                epoch_loss += loss.cpu().detach().numpy()
                loss.backward()
                optimizer.step()
                
            all_loss[epoch] = epoch_loss / (i+1)
            fig, ax = plt.subplots(1, 1, figsize = (20, 10))
            ax.plot([i for i in range(1, num_epochs+1)][:epoch+1], all_loss[:epoch+1], c = "dodgerblue")
            ax.set_xlim(1, num_epochs)
            plt.tight_layout()
            plt.savefig("foo.png")
            plt.close()
            
        # Extract
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
        
        np.save("data_extracted/kinetics400/weighted_kinetics400_%s_%s" % (modality, dataset_names[dataset_id]), all_extracted)
    np.save("data_extracted/kinetics400/weighted_kinetics400_y_%s" % dataset_names[dataset_id], y_extract)
    np.save("data_extracted/kinetics400/weighted_kinetics400_ids_%s" % dataset_names[dataset_id], ids_extract)
        
    print("\n")