"""
Using deep extractors to extract features from each predefined dataset.
Douczac ekstraktory
"""

import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from sentence_transformers import SentenceTransformer
from torch import from_numpy
from tqdm import tqdm


# Batch size for extraction processing
batch_size = 2000

# load data
all_genres = np.load("data_npy/mmIMDb_genres.npy")
all_genres = all_genres[:, 0]

datasets = [
    ["Horror", "Romance"],
    ["Crime", "Documentary", "Fantasy", "Sci-Fi"],
    ["Animation", "Biography", "History", "Music", "War"],
    ["Film-Noir", "Musical", "News", "Short", "Sport", "Western"],
    all_genres,
]

for dataset_id, dataset in tqdm(enumerate(datasets)):
    if dataset_id == (len(datasets)-1):
        dataset_name = "all"
    else:
        dataset_name = "".join([genre[0] for genre in dataset])

    X_img = np.load("data_npy/mmIMDb/mmIMDb_%s_img.npy" % dataset_name)
    X_txt = np.load("data_npy/mmIMDb/mmIMDb_%s_txt.npy" % dataset_name)

    """
    Extraction from images
    """
    print("Extracting IMG from %s!" % dataset_name)
    print(X_img.shape)
    current_sample = 0
    batch_extracted = []
    while current_sample < X_img.shape[0]:
        X_img_batch = X_img[current_sample:current_sample+batch_size]
        # Preprocess images for extraction
        weights = ResNet18_Weights.IMAGENET1K_V1
        preprocess = weights.transforms()
        X_img_batch_transformed = preprocess(from_numpy(np.moveaxis(X_img_batch, 3, 1)))

        # extractor model
        model = resnet18(weights=weights)
        model.eval()
        # train_nodes, eval_nodes = get_graph_node_names(model)
        # print(train_nodes)
        # print(eval_nodes)
        return_nodes = {
            'flatten': 'extracted_flatten',
        }
        extractor = create_feature_extractor(model, return_nodes=return_nodes)
        X_img_batch_extracted = extractor(X_img_batch_transformed)["extracted_flatten"].detach().numpy()
        batch_extracted.append(X_img_batch_extracted)
        current_sample += X_img_batch.shape[0]
        # print("CURRENT SAMPLE: %i" % current_sample)
        # print("Extracted batch")
        # print(X_img_batch_extracted.shape)
    X_img_extracted = np.concatenate(batch_extracted, axis=0)
    print("IMG from %s extracted!" % dataset_name)
    print(X_img_extracted.shape)
    np.save("data_extracted/mmIMDb/mmIMDb_%s_img" % dataset_name, X_img_extracted)

    """
    Text to embeddings
    """
    print("Extracting TXT from %s!" % dataset_name)
    print(X_txt.shape)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    current_sample = 0
    batch_embeddings = []
    while current_sample < X_txt.shape[0]:
        X_txt_batch = X_txt[current_sample:current_sample+batch_size]
        embeddings = embedder.encode(X_txt_batch)
        batch_embeddings.append(embeddings)
        current_sample += X_txt_batch.shape[0]
        # print("CURRENT SAMPLE: %i" % current_sample)

    corpus_embeddings = np.concatenate(batch_embeddings, axis=0)
    print("TXT from %s extracted!" % dataset_name)
    print(corpus_embeddings.shape)
    np.save("data_extracted/mmIMDb/mmIMDb_%s_txt" % dataset_name, corpus_embeddings)