import numpy as np
import umap

features = np.load("features.npy")
emb = umap.UMAP().fit_transform(features)
np.save("embeddings.npy", emb)
# note: Ignore Numba Performance Warnings