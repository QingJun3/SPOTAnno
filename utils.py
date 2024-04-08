import random
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import ot
import os
from sklearn.metrics import classification_report, accuracy_score
from easydl import AccuracyCounter
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment as linear_assignment

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
    torch.use_deterministic_algorithms(True)

class MemoryQueue(nn.Module):
    def __init__(self, feat_dim, batchsize, n_batch, device, T=0.05):
        super(MemoryQueue, self).__init__()
        self.feat_dim = feat_dim
        self.batchsize = batchsize
        self.T = T
        self.device = device

        # init memory queue
        self.queue_size = self.batchsize * n_batch
        self.register_buffer('mem_feat', torch.zeros(self.queue_size, feat_dim))
        self.register_buffer('mem_id', torch.zeros((self.queue_size), dtype=int))
        self.mem_feat = self.mem_feat.to(self.device)
        self.mem_id = self.mem_id.to(self.device)

        # write pointer
        self.next_write = 0

    def forward(self, x):
        """
        obtain similarity between x and the features stored in memory queue
        """
        out = torch.mm(x, self.mem_feat.t()) / self.T
        return out

    def get_nearest_neighbor(self, anchors, id_anchors=None):
        """
        get anchors' nearest neighbor in memory queue 
        """
        # compute similarity first
        feat_mat = self.forward(anchors)

        # assign the similarity between features of the same sample with -1/T
        if id_anchors is not None:
            A = id_anchors.reshape(-1, 1).repeat(1, self.mem_id.size(0))
            B = self.mem_id.reshape(1, -1).repeat(id_anchors.size(0), 1)
            mask = torch.eq(A, B)
            id_mask = torch.nonzero(mask)
            temp = id_mask[:,1]
            feat_mat[:, temp] = -1 / self.T

        # obtain neighbor's similarity value and corresponding feature
        values, indices = torch.max(feat_mat, 1)
        nearest_feat = torch.zeros((anchors.size(0), self.feat_dim)).to(self.device)
        for i in range(anchors.size(0)):
            nearest_feat[i] = self.mem_feat[indices[i],:]
        return values, nearest_feat

    def update_queue(self, features, ids):
        """
        update memory queue
        """
        w_ids = torch.arange(self.next_write, self.next_write+self.batchsize).to(self.device)
        self.mem_feat.index_copy_(0, w_ids, features.data)
        self.mem_id.index_copy_(0, w_ids, ids.data)
        self.mem_feat = F.normalize(self.mem_feat)

        # update write pointer
        self.next_write += self.batchsize
        if self.next_write == self.queue_size:
            self.next_write = 0

    def random_sample(self, size):
        """
        sample some features from memory queue randomly
        """ 
        id_t = torch.floor(torch.rand(size) * self.mem_feat.size(0)).long().to(self.device)
        sample_feat = self.mem_feat[id_t]
        return sample_feat


class ResultsCalculator(object):
    """
    calculate final results (including overall acc, seen acc, novel acc)
    """
    def __init__(self, classes_set, label, predict_label, uniformed_index=None):
        if uniformed_index is None:
            uniformed_index = max(classes_set['source_classes']) + 1
        self.overall_accs, self.overall_acc_aver = self._get_overall_acc(classes_set['source_classes'], label, predict_label, uniformed_index)
        common_accs = dict()
        for index in classes_set['common_classes']:
            common_accs[index] = self.overall_accs[index]
        self.common_acc_aver = np.mean(list(common_accs.values()))
        self.tp_acc = self.overall_accs[uniformed_index]

        # label(seen + novel)
        unif_label = self._tplabel_unif(classes_set['source_classes'], label, uniformed_index)
        self.report_1 = classification_report(unif_label, predict_label, output_dict=True)
        self.overall_acc = accuracy_score(unif_label, predict_label)
        
        # label(seen)
        index_list = []
        for i, x in enumerate(unif_label):
            if x < uniformed_index:
                index_list.append(i)
        
        unif_label = unif_label[index_list]
        predict_label = predict_label[index_list]
        self.report_2 = classification_report(unif_label, predict_label, output_dict=True)
        self.common_acc = accuracy_score(unif_label, predict_label)
        
    def _get_overall_acc(self,source_classes, label, predict_label, uniformed_index):
        unif_label = self._tplabel_unif(source_classes, label, uniformed_index)
        overall_accs, overall_acc_aver = self._recall_acc_cal(source_classes+[uniformed_index], predict_label, unif_label)
        return overall_accs, overall_acc_aver

    def _recall_acc_cal(self, classes, predict_label, label):
        counters = {class_label:AccuracyCounter() for class_label in classes}
        for (each_predict_label, each_label) in zip(predict_label, label):
            if each_label in classes:
                counters[each_label].Ntotal += 1.0
                if each_predict_label == each_label:
                    counters[each_label].Ncorrect += 1.0
        recall_accs = {i:counters[i].reportAccuracy() for i in counters.keys() \
                                                        if not np.isnan(counters[i].reportAccuracy())}
        recall_acc_aver = np.mean(list(recall_accs.values()))
        return recall_accs, recall_acc_aver
    
    def _tplabel_unif(self, source_classes, label, uniformed_index):
        uniform_tp = label.copy()
        for i in range(len(label)):
            if label[i] not in source_classes:
                uniform_tp[i] = uniformed_index
        return uniform_tp

    def _nmi_cal(self, label, proto_pred):
        nmi = normalized_mutual_info_score(label, proto_pred)
        return nmi

    def _hungarian_matching(self, label, proto_pred):
        assert proto_pred.size == label.size
        matrix_size = max(len(set(proto_pred)), len(set(label)))
        matrix = np.zeros((matrix_size, matrix_size), dtype=np.int64)
        for i in range(proto_pred.size):
            matrix[proto_pred[i], label[i]] += 1
        mat_index = linear_assignment(matrix.max() - matrix)
        return matrix,mat_index
    
    def _clutser_acc_cal(self, common_classes, tp_classes, label, proto_pred):
        classes = []
        classes.extend(common_classes)
        classes.extend(tp_classes)
        counters = {class_label:AccuracyCounter() for class_label in classes}
        for index in label:
            counters[index].Ntotal += 1
        index2label_map = {i:classes[i] for i in range(len(classes))}
        label2index_map = {classes[i]:i for i in range(len(classes))}
        transform_label = np.zeros_like(label)
        for i in range(len(label)):
            transform_label[i] = label2index_map[label[i]]
        matrix,mat_index = self._hungarian_matching(transform_label, proto_pred)
        sum_Ncorrect = 0
        for (i,j) in zip(mat_index[0],mat_index[1]):
            if index2label_map.get(j) is not None:
                counters[index2label_map[j]].Ncorrect = matrix[i,j]
                sum_Ncorrect += matrix[i,j]
        
        self.traditional_acc = sum_Ncorrect * 1.0 / len(proto_pred)

        self.overall_accs = {i:counters[i].reportAccuracy() for i in counters.keys() \
                                                        if not np.isnan(counters[i].reportAccuracy())}
        self.overall_acc = np.mean(list(self.overall_accs.values()))
        
        common_accs = dict()
        for index in common_classes:
            common_accs[index] = self.overall_accs[index]
        self.common_acc = np.mean(list(common_accs.values()))

        tp_accs = dict()
        for index in tp_classes:
            if index in self.overall_accs:
                tp_accs[index] = self.overall_accs[index]
        self.tp_acc = np.mean(list(tp_accs.values()))
    
def entropy(p):
    p = F.softmax(p)
    return -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))