import itertools as its
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# from sklearn.model_selection import GridSearchCV as GS


def relation_matrix_torch_test(vec1, vec2, e=0):
    dist_matrix = torch.cdist(vec1, vec2, p=1)
    if e == -1:
        return (dist_matrix < 1e-6).float()
    relation_matrix = 1 - dist_matrix
    relation_matrix[dist_matrix > e] = 0
    return relation_matrix

import itertools as its
import numpy as np
import torch
torch.set_default_dtype(torch.float32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def merge_set(groups, l, r):
    for idx, i in enumerate(groups):
        if l in i:
            left_idx = idx
        if r in i:
            right_idx = idx
    if left_idx==right_idx:
        return groups
    # new = np.concatenate([groups[left_idx], groups[right_idx]], axis=1)
    l_all, r_all= groups[left_idx], groups[right_idx]
    new = l_all+r_all
    new.sort()
    groups.remove(l_all)
    groups.remove(r_all)
    return sorted([new] + groups)

class GREAD(object):
    def __init__(self, data, nominals, lamb=0.5):
        n, m = data.shape
        self.nominals = nominals
        self.data = torch.from_numpy(data).float().T.reshape(m, n, 1)
        self.lambs = np.array([lamb] * m, dtype=np.float32())
        self.lamb = lamb
        self.lambs[nominals] = -1
        self.fmi = self.NFMI()
        self.attr_P = None
        print('\tLamb={}, NFMI: mean={:.2f}, median={:.2f}, max={:.2f}'.format(lamb, self.fmi.mean(),np.median(self.fmi), self.fmi.max()),end='\t')

    def relation_matrix_torch(self, vec1, vec2, lamb):
        dist_matrix = torch.cdist(vec1, vec2, p=1)
        if lamb < -0.1:
            return (dist_matrix < 1e-6).float()
        e = np.quantile(dist_matrix.flatten(), lamb)
        dist_matrix *= -1
        dist_matrix += 1
        dist_matrix[dist_matrix < 1 - e] = 0
        return dist_matrix

    def NFMI(self):
        m, n, _ = self.data.shape
        mat_rel_singleton = torch.zeros((m, n, n), dtype=torch.float32)
        for j in range(m):
            mat = self.data[j]
            mat_rel_singleton[j] = self.relation_matrix_torch(mat, mat, lamb=self.lambs[j])
        FE_singleton = -torch.mean(torch.log2(mat_rel_singleton.mean(dim=1)), dim=1)
        # print('Entropy of attribues:',FE_singleton)

        fuzzy_mutual_info = []
        for i,j in its.combinations(np.arange(m), 2):
            FE_ij = -torch.log2(torch.minimum(mat_rel_singleton[i], mat_rel_singleton[j]).mean(0)).mean(0)
            fmi = (FE_singleton[i] + FE_singleton[j] - FE_ij) / torch.sqrt(FE_singleton[i] * FE_singleton[j])
            # print(f"FE(X): {FE_singleton[i]}, FE(Y): {FE_singleton[j]}, FE(X,Y):{FE_ij}, NFMI:{fmi}")
            fuzzy_mutual_info.append(fmi)
            assert not np.isnan(fmi), 'Some attribute is a constant'
        return np.array(fuzzy_mutual_info)

    def attribute_partition(self, threshold=0.5):
        m, n, _ = self.data.shape
        mi = self.fmi
        subgroups = np.arange(m).reshape(-1, 1).tolist()
        idx = 0
        for left, right in its.combinations(np.arange(m), 2):
            if mi[idx] > threshold:
                subgroups = merge_set(subgroups, left, right)
            idx += 1
        # print('\t\t\tThreshold={} Subfeatures:{}'.format(threshold, subgroups))
        print('\t\t\tSubfeatures:{}'.format(subgroups))

        if self.attr_P is not None:
            if self.attr_P == subgroups:
                return True

        self.attr_P = subgroups
        # print(self.attr_P, subgroups)
        return False

    def make_fuzzy_relative_entropy(self):
        m, n, _ = self.data.shape
        n_bins = len(self.attr_P)
        dist_rel_mat = torch.zeros((n_bins, n, n), dtype=torch.float32)
        nonimals = np.arange(m)[self.nominals]
        for idx, bin in enumerate(self.attr_P):
            if len(bin) == 1:
                temp = torch.cdist(self.data[bin], self.data[bin], p=1)
                if bin in nonimals:
                    temp = (temp > 1e-6).float()
                else:
                    e = np.quantile(temp.flatten(), self.lamb)
                    temp[temp > e] = 1
            else:
                temp = torch.cdist(self.data[bin], self.data[bin], p=1)
                for idx_, i in enumerate(bin):
                    if i in nonimals:
                        temp[idx_] = temp[idx_] > 1e-6
                temp = temp.mean(axis=0)
                e = np.quantile(temp.flatten(), self.lamb)
                temp[temp > e] = 1

            dist_rel_mat[idx] = 1 - temp
        # print('\t\t\tRelation: mean={:.2f}'.format(dist_rel_mat.mean()))

        del temp, nonimals

        FE_B_x = torch.zeros((n, n_bins), dtype=torch.float32)
        # for j in range(n_bins):
        #     relation_matrix_sum_x = torch.tile(dist_rel_mat[j].sum(dim=0), (n, 1)) - dist_rel_mat[j]
        #     relation_matrix_sum_x = torch.tril(relation_matrix_sum_x, diagonal=-1)[:, :-1] + torch.triu(relation_matrix_sum_x, diagonal=1)[:, 1:]
        #     FE_B_x[:, j] = -torch.sum(torch.log2(relation_matrix_sum_x / (n - 1)), dim=1) / (n - 1)
        # del relation_matrix_sum_x

        diag_idx = np.arange(n)
        for j in range(n_bins):
            relation_matrix_mean_x = (torch.tile(dist_rel_mat[j].sum(dim=0), (n, 1)) - dist_rel_mat[j]) / (n-1)
            relation_matrix_mean_x[diag_idx, diag_idx] = 1 # Equals to remove the value of each x as torch.log2(1) = 0
            FE_B_x[:, j] = -torch.sum(torch.log2(relation_matrix_mean_x), dim=1) / (n - 1)
            del relation_matrix_mean_x

        self.card_mean = dist_rel_mat.mean(dim=1)
        del dist_rel_mat

        FE_B = -torch.mean(torch.log2(self.card_mean), dim=1)
        self.FRE = (FE_B_x + 1e-6) / (FE_B + 1e-6) + 1/n

    def predict_score(self):
        self.make_fuzzy_relative_entropy()
        OD = 1 - (torch.sqrt(self.card_mean) * self.FRE.T).mean(axis=0) # 24个数据集平均AUC=0.912, PR=0.597
        return OD
