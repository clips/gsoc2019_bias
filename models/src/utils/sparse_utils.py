import numpy as np
import torch
from scipy.sparse import csr_matrix

def save_sparse_csr(filename, matrix):
    np.savez(filename, data=matrix.data, indices=matrix.indices,
             indptr=matrix.indptr, shape=matrix.shape)

def load_sparse_csr(filename):
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

def convert_sparse_matrix_to_tensor(x):
    coo = x.tocoo()

    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(np.int64(indices)).long()
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()