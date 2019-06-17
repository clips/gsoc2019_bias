import numpy as np
import torch


def convert_sparse_matrix_to_sparse_tensor(x):
    coo = x.tocoo()

    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(np.int64(indices)).long()
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()