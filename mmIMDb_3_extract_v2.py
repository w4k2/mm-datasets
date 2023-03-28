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
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AlbertTokenizer, pipeline, PreTrainedTokenizer
import torch.optim as optim
import matplotlib.pyplot as plt


# load data
all_genres = np.load("data_npy/mmIMDb_genres.npy")
all_genres = all_genres[:, 0]

datasets = [
    ["Horror", "Romance"],
    ["Crime", "Documentary", "Fantasy", "Sci-Fi"],
    ["Animation", "Biography", "History", "Music", "War"],
    ["Film-Noir", "Musical", "News", "Short", "Sport", "Western"],
    ['Action', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Horror', 'Mystery', 'Romance', 'Sci-Fi'],
]

for dataset_id, dataset in tqdm(enumerate(datasets)):
    dataset_name = "".join([genre[0] for genre in dataset])
    print(dataset_name)
    X_img = np.load("data_npy/mmIMDb/mmIMDb_%s_img.npy" % dataset_name)
    X_txt = np.load("data_npy/mmIMDb/mmIMDb_%s_txt.npy" % dataset_name)
    y = np.load("data_npy/mmIMDb/mmIMDb_%s_y.npy" % dataset_name).astype(int)
    
    print(np.unique(y, return_counts=True))
    X_img_train, X_img_extract, X_txt_train, X_txt_extract, y_train, y_extract = train_test_split(
            X_img, X_txt, y, test_size=.8, random_state=1410, stratify=y, shuffle=True)
    
    """
    Extraction from images
    """
    
    # # Training
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
    criterion = nn.CrossEntropyLoss(weight=weights)

    dataset_ready = TensorDataset(from_numpy(np.moveaxis(X_img_train, 3, 1)).float(), from_numpy(y_train).long())
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
    X_img_extract = from_numpy(np.moveaxis(X_img_extract, 3, 1)).float()
    model.eval()

    all_extracted = []
    current_sample = 0
    batch_size = 500
    print("EXTRACION!")
    while current_sample < X_img_extract.shape[0]:
        print("Batch %i:%i" % (current_sample, current_sample+batch_size))
        X_img_extract_batch = X_img_extract[current_sample:current_sample+batch_size]
        return_nodes = {
            'flatten': 'extracted_flatten',
        }
        extractor = create_feature_extractor(model, return_nodes=return_nodes)
        X_img_batch_extracted = extractor(X_img_extract_batch.to(device))["extracted_flatten"].cpu().detach().numpy()
        
        
        all_extracted.append(X_img_batch_extracted)
        current_sample += X_img_extract_batch.shape[0]
        
    all_extracted = np.vstack(tuple(all_extracted))
    print("IMG from %s extracted!" % dataset_name)
    print(all_extracted.shape)
    np.save("data_extracted/mmIMDb/mmIMDb_%s_img" % dataset_name, all_extracted)


    """
    Text to embeddings
    """
    
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v1")
    
    encoding = tokenizer(list(X_txt_train), truncation=True, return_tensors='pt', padding=True, max_length=512)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    
    dataset_ready = TensorDataset(input_ids, attention_mask, from_numpy(y_train).long())
    data_loader = DataLoader(dataset_ready, batch_size=batch_size, shuffle=True)
    
    model = AutoModelForSequenceClassification.from_pretrained("albert-base-v1", num_labels=num_classes)
    
    device = torch.device("mps")
    model = model.to(device)

    params_to_update = model.parameters()
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    _, counts = np.unique(y_train, return_counts=True)
    weights = from_numpy(np.array([1-(count/np.sum(counts)) for count in counts])).float().to(device)
    print(weights)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model.train()
    print("TRAINING!")
    all_loss = np.zeros(num_epochs)
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0            
        for i, batch in enumerate(data_loader, 0):
            # print("ZROBILEM BATCH!")
            inputs, attention, labels = batch
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to(device), attention_mask=attention.to(device))
            loss = criterion(outputs.logits.to(device), labels.to(device))
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
    
    model.eval()
    model.save_pretrained("models/%s" % dataset_name)
    tokenizer.save_pretrained("models/%s" % dataset_name)

    all_extracted = []
    current_sample = 0
    batch_size = 500
    print("EXTRACION!")
    
    feature_extractor = pipeline("feature-extraction",framework="pt",model="models/%s" % dataset_name)
    
    while current_sample < X_txt_extract.shape[0]:
        print("Batch %i:%i" % (current_sample, current_sample+batch_size))
        X_txt_extract_batch = X_txt_extract[current_sample:current_sample+batch_size]
        
        extracted = feature_extractor(list(X_txt_extract_batch),return_tensors = "pt", truncation=True, max_length=512)
        extracted = np.array([extract.numpy().mean(axis=(0,1)) for extract in extracted])
        
        all_extracted.append(extracted)
        current_sample += X_txt_extract_batch.shape[0]
    
    all_extracted = np.vstack(tuple(all_extracted))
    print("IMG from %s extracted!" % dataset_name)
    print(all_extracted.shape)
    np.save("data_extracted/mmIMDb/mmIMDb_%s_txt" % dataset_name, all_extracted)
    
    print("y from %s extracted!" % dataset_name)
    print(y_extract.shape)
    np.save("data_extracted/mmIMDb/mmIMDb_%s_y" % dataset_name, y_extract)