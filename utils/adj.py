import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
import torch


def build_knn_adj(user_pos_items, item_feats, topk_per_user):
    """Ablation3: KNN to generate modality user-item adjacency matrix"""
    user_proto = np.array([
        item_feats[items].mean(axis=0) if len(items) > 0 else np.zeros(item_feats.shape[1])
        for items in user_pos_items
    ])  # shape=(user_num, feat_dim)

    sim = cosine_similarity(user_proto, item_feats)  # (user_num, item_num)

    u_list, i_list, vals = [], [], []
    for u in range(sim.shape[0]):
        idx = np.argsort(-sim[u])[:topk_per_user]
        u_list.extend([u] * topk_per_user)
        i_list.extend(idx.tolist())
        vals.extend([1.0] * topk_per_user)
    return np.array(u_list), np.array(i_list), np.array(vals)


def norm_adj(mat: coo_matrix): 
    """
    Normalize a sparse adjacency matrix using the symmetric normalization method D^(-1/2) * A * D^(-1/2).

    Args:
        mat (scipy.sparse.coo_matrix): (node_num, node_num)
    """
    csr_mat = mat.tocsr()  #* for faster computation
    # matrix's element has been set to 1.0, so degree is the number of non-zero elements in each row
    degree = np.asarray(csr_mat.sum(axis=1)).squeeze()
    dInvSqrt = np.where(degree > 0, degree**(-0.5), 0)
    dInvSqrtMat = sp.diags(dInvSqrt, offsets=0, format='csr')
    normalized_mat: csr_matrix = dInvSqrtMat @ mat @ dInvSqrtMat
    return normalized_mat.tocoo()


def torch_sparse_adj(mat: coo_matrix, rows: int, cols: int, device: torch.device):
    """
    Construct a sparse bipartite adjacency matrix and convert to torch sparse tensor.

    Args:
        mat (scipy.sparse.coo_matrix): (user_num, item_num)

    Returns:
        out (torch.sparse_coo_tensor): (node_num, node_num)
    """
    #* build a sparse bipartite adjacency matrix and normalize it
    a = csr_matrix((rows, rows))
    b = csr_matrix((cols, cols))
    adj_mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])]).tocoo() # (node_num, node_num), node_num = user_num + item_num
    adj_mat = (adj_mat != 0) * 1.0 # convert to binary matrix (data is float)
    adj_mat = (adj_mat + sp.eye(adj_mat.shape[0])) * 1.0  # set diagonal to 1 (self connection)
    adj_mat = norm_adj(adj_mat)

    #* make cuda tensor
    idxs = torch.from_numpy(np.vstack([adj_mat.row, adj_mat.col]).astype(np.int64))
    vals = torch.from_numpy(adj_mat.data.astype(np.float32))
    if adj_mat.shape is None:
        raise ValueError("adj_mat has no defined shape. Ensure it is a valid sparse matrix.")
    shape = torch.Size(adj_mat.shape)
    return torch.sparse_coo_tensor(idxs, vals, shape, device=device)  # torch.Size([node_num, node_num])