import faiss
import numpy as np

class ImpRAGFAISSIndex:
    """
    FAISS Index wrapper using Flat Inner Product (dot product) similarity.
    Matches the similarity score formula in Section 3.1: s(q, p) = E_q * E_p
    """
    def __init__(self, dimension):
        self.dimension = dimension
        # Use IndexFlatIP for Inner Product (Dot Product) search
        self.index = faiss.IndexFlatIP(dimension)
        self.mean_vector = None
        self.query_mean = None
        
    def add_embeddings(self, embeddings):
        """
        embeddings: numpy array of shape [num_items, dimension] and type float32.
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
            
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            
        assert embeddings.shape[1] == self.dimension, f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
        
        # Calculate mean vector for centering
        self.mean_vector = embeddings.mean(axis=0, keepdims=True)
        
        # Center and normalize embeddings
        embeddings_centered = embeddings - self.mean_vector
        norms = np.linalg.norm(embeddings_centered, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        embeddings_centered /= norms
        
        self.index.add(embeddings_centered)
        
    def search(self, query_embeddings, k=5):
        """
        query_embeddings: numpy array of shape [batch_size, dimension] and type float32.
        Returns:
            distances: numpy array of shape [batch_size, k] (inner product scores)
            indices: numpy array of shape [batch_size, k] (passage index integers)
        """
        if not isinstance(query_embeddings, np.ndarray):
            query_embeddings = np.array(query_embeddings, dtype=np.float32)
            
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
            
        assert query_embeddings.shape[1] == self.dimension, f"Query dimension mismatch: expected {self.dimension}, got {query_embeddings.shape[1]}"
        
        # Center using query_mean if available, fallback to mean_vector (document mean)
        q_mean = self.query_mean if self.query_mean is not None else self.mean_vector
        if q_mean is not None:
            query_embeddings = query_embeddings - q_mean
            norms = np.linalg.norm(query_embeddings, axis=-1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            query_embeddings /= norms
            
        # search returns (distances, indices)
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices
        
    def save(self, filepath):
        """
        Saves the FAISS index to a file on disk, along with the mean vectors.
        """
        faiss.write_index(self.index, filepath)
        if self.mean_vector is not None:
            np.save(filepath + ".mean.npy", self.mean_vector)
        if self.query_mean is not None:
            np.save(filepath + ".query_mean.npy", self.query_mean)
        
    @classmethod
    def load(cls, filepath):
        """
        Loads a FAISS index and its mean vectors from disk.
        """
        import os
        index = faiss.read_index(filepath)
        obj = cls(index.d)
        obj.index = index
        
        mean_path = filepath + ".mean.npy"
        if os.path.exists(mean_path):
            obj.mean_vector = np.load(mean_path)
        else:
            obj.mean_vector = None
            
        query_mean_path = filepath + ".query_mean.npy"
        if os.path.exists(query_mean_path):
            obj.query_mean = np.load(query_mean_path)
        else:
            obj.query_mean = None
        return obj
        
    def get_num_vectors(self):
        """
        Returns the number of vectors stored in the index.
        """
        return self.index.ntotal
