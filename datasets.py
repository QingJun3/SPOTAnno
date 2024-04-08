import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import scanpy as sc

# Dataset
class ST_Dataset(Dataset):
    def __init__(self, data, label, x, y, return_id=True):
        self.datas = np.array(data, dtype='float32')
        self.labels = label

        self.x = np.array(x, dtype='float32')
        self.y = np.array(y, dtype='float32')
        # normalize
        scale = max(self.x.max() - self.x.min(), self.y.max() - self.y.min())
        self.x = (self.x - self.x.min()) / scale
        self.y = (self.y - self.y.min()) / scale

        self.return_id = return_id

    def __getitem__(self, index):
        if self.return_id:
            return self.datas[index], self.labels[index], self.x[index], self.y[index], index
        else:
            return self.datas[index], self.labels[index], self.x[index], self.y[index]
    def __len__(self):
        return self.datas.shape[0]

def make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, class_balance, batch_size, num_workers):
    
    # split common | train_private | test_private
    common_y = np.intersect1d(train_y, test_y)
    train_private_y = np.setdiff1d(train_y, common_y)
    test_private_y = np.setdiff1d(test_y, common_y)

    print(f"common: {common_y}")
    print(f"train private: {train_private_y}")
    print(f"test_private: {test_private_y}")

    # to digit
    cell_type_dict = {}
    inverse_dict = {}
    cnt = 0

    for y in common_y:
        cell_type_dict[y] = cnt
        inverse_dict[cnt] = y
        cnt += 1

    for y in train_private_y:
        cell_type_dict[y] = cnt
        inverse_dict[cnt] = y
        cnt += 1

    for y in test_private_y:
        cell_type_dict[y] = cnt
        inverse_dict[cnt] = y
        cnt += 1

    train_y = np.array([cell_type_dict[x] for x in train_y])
    test_y = np.array([cell_type_dict[x] for x in test_y])

    # make classes set
    a, b, c = common_y.shape[0], train_private_y.shape[0], test_private_y.shape[0]
    common_classes = [i for i in range(a)]
    source_private_classes = [i + a for i in range(b)]
    target_private_classes = [i + a + b for i in range(c)]

    source_classes = common_classes + source_private_classes
    target_classes = common_classes + target_private_classes

    # target-private label
    tp_classes = sorted(set(target_classes) - set(source_classes))
    # source-private label
    sp_classes = sorted(set(source_classes) - set(target_classes))
    # common label
    common_classes = sorted(set(source_classes) - set(sp_classes))

    classes_set = {
    'source_classes': source_classes,
    'target_classes': target_classes,
    'tp_classes': tp_classes,
    'sp_classes': sp_classes,
    'common_classes': common_classes
    }

    # make dataset and dataloader
    uniformed_index = len(classes_set['source_classes'])

    source_train_ds = ST_Dataset(train_X, train_y, labeled_pos[0], labeled_pos[1])
    target_train_ds = ST_Dataset(test_X, test_y, unlabeled_pos[0], unlabeled_pos[1])

    source_test_ds = ST_Dataset(train_X, train_y, labeled_pos[0], labeled_pos[1], return_id=False)
    target_test_ds = ST_Dataset(test_X, test_y, unlabeled_pos[0], unlabeled_pos[1], return_id=False)

    # balanced sampler for source train
    classes = source_train_ds.labels
    freq = Counter(classes)
    class_weight = {x : 1.0 / freq[x] if class_balance else 1.0 for x in freq}

    source_weights = [class_weight[x] for x in source_train_ds.labels]
    sampler = WeightedRandomSampler(source_weights, len(source_train_ds.labels))

    source_train_dl = DataLoader(dataset=source_train_ds, batch_size=batch_size,
                             sampler=sampler, num_workers=num_workers, drop_last=True)
    
    target_train_dl = DataLoader(dataset=target_train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=True)
    target_test_dl = DataLoader(dataset=target_test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)

    # for memory queue init
    target_initMQ_dl = DataLoader(dataset=target_train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=True)
    # for tsne feature visualization
    source_test_dl = DataLoader(dataset=source_test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    
    return classes_set, train_X.shape[1], source_train_dl, target_train_dl, target_test_dl, target_initMQ_dl, source_test_dl

def load_Hubmap_CL_intra_data(args):

    train_df = pd.read_csv("data/Hubmap_CL_intra/train.csv")
    test_df = pd.read_csv("data/Hubmap_CL_intra/test.csv")

    train_X = train_df.iloc[:, 1:49].values
    test_X = test_df.iloc[:, 1:49].values
    train_y = train_df['cell_type_A']
    test_y = test_df['cell_type_A']

    labeled_pos = train_df.iloc[:, -6:-4].values.T
    unlabeled_pos = test_df.iloc[:, -6:-4].values.T

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_Hubmap_SB_intra_data(args):

    train_df = pd.read_csv("data/Hubmap_SB_intra/train.csv")
    test_df = pd.read_csv("data/Hubmap_SB_intra/test.csv")

    train_X = train_df.iloc[:, 1:49].values
    test_X = test_df.iloc[:, 1:49].values
    train_y = train_df['cell_type_A']
    test_y = test_df['cell_type_A']

    labeled_pos = train_df.iloc[:, -6:-4].values.T
    unlabeled_pos = test_df.iloc[:, -6:-4].values.T

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_Hyp_intra_data(args):

    train_df = pd.read_csv("data/Hyp_intra/train.csv")
    test_df = pd.read_csv("data/Hyp_intra/test.csv")

    train_X = train_df.iloc[:, 10:].values
    test_X = test_df.iloc[:, 10:].values
    train_y = train_df["Cell_class"].values
    test_y = test_df["Cell_class"].values

    labeled_pos = train_df.iloc[:, 6:8].values.T
    unlabeled_pos = test_df.iloc[:, 6:8].values.T

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_Spe_Diabetes_intra_data(args):

    train_adata = sc.read_h5ad("data/Spe_Diabetes_intra/train.h5ad")
    test_adata = sc.read_h5ad("data/Spe_Diabetes_intra/test.h5ad")

    train_X = train_adata.X
    test_X = test_adata.X
    train_y = train_adata.obs['CellType'].values
    test_y = test_adata.obs['CellType'].values

    labeled_pos = []
    labeled_pos.append(train_adata.obs['pos_x'].values)
    labeled_pos.append(train_adata.obs['pos_y'].values)
    unlabeled_pos = []
    unlabeled_pos.append(test_adata.obs['pos_x'].values)
    unlabeled_pos.append(test_adata.obs['pos_y'].values)

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_Spe_WT_intra_data(args):

    train_adata = sc.read_h5ad("data/Spe_WT_intra/train.h5ad")
    test_adata = sc.read_h5ad("data/Spe_WT_intra/test.h5ad")

    train_X = train_adata.X
    test_X = test_adata.X
    train_y = train_adata.obs['CellType'].values
    test_y = test_adata.obs['CellType'].values

    labeled_pos = []
    labeled_pos.append(train_adata.obs['pos_x'].values)
    labeled_pos.append(train_adata.obs['pos_y'].values)
    unlabeled_pos = []
    unlabeled_pos.append(test_adata.obs['pos_x'].values)
    unlabeled_pos.append(test_adata.obs['pos_y'].values)

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)


def load_Hubmap_CL_cross_data(args):

    train_df = pd.read_csv("data/Hubmap_CL_cross/train.csv")
    test_df = pd.read_csv("data/Hubmap_CL_cross/test.csv")

    train_X = train_df.iloc[:, 1:49].values
    test_X = test_df.iloc[:, 1:49].values
    train_y = train_df['cell_type_A']
    test_y = test_df['cell_type_A']

    labeled_pos = train_df.iloc[:, -6:-4].values.T
    unlabeled_pos = test_df.iloc[:, -6:-4].values.T

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_Hubmap_SB_cross_data(args):

    train_df = pd.read_csv("data/Hubmap_SB_cross/train.csv")
    test_df = pd.read_csv("data/Hubmap_SB_cross/test.csv")

    train_X = train_df.iloc[:, 1:49].values
    test_X = test_df.iloc[:, 1:49].values
    train_y = train_df['cell_type_A']
    test_y = test_df['cell_type_A']

    labeled_pos = train_df.iloc[:, -6:-4].values.T
    unlabeled_pos = test_df.iloc[:, -6:-4].values.T

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_Spe_Diabetes_cross_data(args):

    train_adata = sc.read_h5ad("data/Spe_Diabetes_cross/train.h5ad")
    test_adata = sc.read_h5ad("data/Spe_Diabetes_cross/test.h5ad")

    train_X = train_adata.X
    test_X = test_adata.X
    train_y = train_adata.obs['CellType'].values
    test_y = test_adata.obs['CellType'].values

    labeled_pos = []
    labeled_pos.append(train_adata.obs['pos_x'].values)
    labeled_pos.append(train_adata.obs['pos_y'].values)
    unlabeled_pos = []
    unlabeled_pos.append(test_adata.obs['pos_x'].values)
    unlabeled_pos.append(test_adata.obs['pos_y'].values)

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_Spe_WT_cross_data(args):

    train_adata = sc.read_h5ad("data/Spe_WT_cross/train.h5ad")
    test_adata = sc.read_h5ad("data/Spe_WT_cross/test.h5ad")

    train_X = train_adata.X
    test_X = test_adata.X
    train_y = train_adata.obs['CellType'].values
    test_y = test_adata.obs['CellType'].values

    labeled_pos = []
    labeled_pos.append(train_adata.obs['pos_x'].values)
    labeled_pos.append(train_adata.obs['pos_y'].values)
    unlabeled_pos = []
    unlabeled_pos.append(test_adata.obs['pos_x'].values)
    unlabeled_pos.append(test_adata.obs['pos_y'].values)

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)