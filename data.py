import pickle
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from utils.conf import Config
import torch
from torch.utils.data import Dataset as torch_dataset
import torch.utils.data as dataloader
import os
from utils.adj import torch_sparse_adj

class DataHandler:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(f"cuda:{self.config.base.gpu}" if torch.cuda.is_available() else "cpu")
        data_dir = os.path.join(self.config.data.dir, self.config.data.name)
        if not os.path.exists(data_dir):
            raise ValueError(f"Unknown dataset: {self.config.data.name}")

        #* all datasets' file names are the same
        self.trainfile = f"{data_dir}/trnMat.pkl"
        self.testfile = f"{data_dir}/tstMat.pkl"

        self.imagefile = f"{data_dir}/image_feat.npy"
        self.textfile = f"{data_dir}/text_feat.npy"

        if self.config.data.name == 'tiktok':  # only tiktok has audio features
            self.audiofile = f"{data_dir}/audio_feat.npy"
        
        #* other delayed initialization are in `LoadData()`

    def load_file(self, filename):
        """
        Load pickle file and convert it to a sparse matrix.
        """
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)

        if not isinstance(ret, coo_matrix): # for multi-modal features (.npy)
            ret = coo_matrix(ret)
        return ret

    def load_feature(self, filename):
        """
        Load multi-modal features from .npy file and convert to torch tensor.
        
        Returns:
            tuple:
                - feats (torch.Tensor): (node_num, feat_dim)
                - feat_dim (int)
        """
        feats: np.ndarray = np.load(filename)
        return torch.tensor(feats, dtype=torch.float, device=self.device), feats.shape[1]

    def load_data(self):
        """
        Load training and testing data, and features.
        """
        train_mat = self.load_file(self.trainfile)
        test_mat = self.load_file(self.testfile)
        self.trainMat = train_mat
        #args.user, args.item = trainMat.shape  # (user_num, item_num)
        self.config.data.user_num, self.config.data.item_num = train_mat.shape  # (user_num, item_num)
        self.torchBiAdj = torch_sparse_adj(train_mat, self.config.data.user_num, self.config.data.item_num, self.device) # (node_num, node_num)

        self.train_data = TrainData(train_mat, self.config)
        self.train_loader: dataloader.DataLoader[TrainData] = dataloader.DataLoader(self.train_data, batch_size=self.config.train.batch, shuffle=True, num_workers=0)
        self.test_data = TestData(test_mat, train_mat)
        self.test_loader: dataloader.DataLoader[TestData] = dataloader.DataLoader(self.test_data, batch_size=self.config.train.batch, shuffle=False, num_workers=0)

        self.image_feats, self.config.data.image_feat_dim = self.load_feature(self.imagefile)
        self.text_feats, self.config.data.text_feat_dim = self.load_feature(self.textfile)
        if self.config.data.name == 'tiktok':
            self.audio_feats, self.config.data.audio_feat_dim = self.load_feature(self.audiofile)

        self.diffusion_data = DiffusionData(torch.tensor(self.trainMat.toarray(), dtype=torch.float, device=self.device), self.config) # .A == .toarray()
        self.diffusion_loader: dataloader.DataLoader[DiffusionData] = dataloader.DataLoader(self.diffusion_data, batch_size=self.config.train.batch, shuffle=True, num_workers=0)
        # Expose user_pos_items to handler
        self.user_pos_items = self.train_data.user_pos_items

    def getUserDegrees(self) -> np.ndarray:
        if not hasattr(self, 'trainMat'):
            raise ValueError("Training matrix not loaded. Please call LoadData() first.")
        user_degrees = np.asarray(self.trainMat.sum(axis=1), dtype=int).squeeze()
        return user_degrees

class TrainData(torch_dataset):
    """Train Dataset (with negative sampling func)"""
    def __init__(self, coomat: coo_matrix, config: Config):
        self.config = config
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok() #* dictionary of keys (row, col) and values (data)
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
        # Construct positive item list for each user
        self.user_pos_items = [[] for _ in range(coomat.shape[0])]
        for u, i in zip(self.rows, self.cols):
            self.user_pos_items[u].append(i)

    def neg_sampling(self):
        """select negative samples for each interaction"""
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                neg_index = np.random.randint(self.config.data.item_num)
                if (u, neg_index) not in self.dokmat:
                    break
            self.negs[i] = neg_index

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        """idx -> (user, pos_item, neg_item)"""
        return self.rows[idx], self.cols[idx], self.negs[idx]

class TestData(torch_dataset):
    def __init__(self, testMat: coo_matrix, trainMat: coo_matrix):
        """
        Test Dateset

        Args:
            testMat (coo_matrix): (user_num, item_num)
            trainMat (coo_matrix): (user_num, item_num)
        """
        self.trainMat_csr: csr_matrix = (trainMat.tocsr() != 0) * 1.0

        test_use_its: list = [None] * testMat.shape[0] # users' interactions in test set
        test_users = set()
        for i in range(len(testMat.data)):
            user_idx = testMat.row[i]
            item_idx = testMat.col[i]
            if test_use_its[user_idx] is None:
                test_use_its[user_idx] = list()
            #* coordinate correspondence
            test_use_its[user_idx].append(item_idx)
            test_users.add(user_idx)
        test_users = np.array(list(test_users))
        self.test_users = test_users
        self.test_user_its = test_use_its

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, idx):
        """get user's its in train set and flatten it"""
        return self.test_users[idx], np.reshape(self.trainMat_csr[self.test_users[idx]].toarray(), [-1])
    
class DiffusionData(torch_dataset):
    def __init__(self, data: torch.Tensor, config: Config):
        self.data = data  # (user_num, item_num)
        self.device = torch.device(f"cuda:{config.base.gpu}" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, index):
        item = self.data[index]
        return item, index
    
    def __len__(self):
        return len(self.data)