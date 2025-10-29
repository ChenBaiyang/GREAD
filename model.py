import torch
import itertools as its
from sklearn.metrics import roc_auc_score
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

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


def relation_matrix_torch(vec, p=0.7):
    dist_rel_mat = torch.cdist(vec, vec, p=1).float()
    if p < -0.1:
        return dist_rel_mat.lt_(1e-6) # equal to: (dist_rel_mat < 1e-6).float()

    n = vec.shape[0]
    k = int(p * n * n)  # k-th minimum value = p-th percentile
    delta = dist_rel_mat.view(-1).kthvalue(k).values # equal to: torch.quantile(dist_rel_mat.view(-1), p)

    dist_rel_mat.mul_(-1).add_(1)
    dist_rel_mat.masked_fill_(dist_rel_mat < 1 - delta, 0)
    return dist_rel_mat


class GREAD(object):
    def __init__(self, data, nominals, p=0.7):
        n, m = data.shape
        self.data = torch.from_numpy(data.astype(np.float32)).T.unsqueeze(-1).to(device)
        self.nominals = nominals
        self.p = p
        self.ps = np.full(m, p, dtype=np.float32())
        self.ps[nominals] = -1
        self.attr_partition = None
        self.n_partitions = None
        self.fmi = self.NFMI_v2() if m*n*n*4/1024/1024/1024 > 23 else self.NFMI_v1()
        print('\tp={}, NFMI: mean={:.2f}, median={:.2f}, max={:.2f}'.format(p, self.fmi.mean(),np.median(self.fmi), self.fmi.max()))#,end='\t')


    def NFMI_v1(self):
        m, n, _ = self.data.shape
        mat_rel_singleton = torch.zeros((m, n, n), dtype=torch.float32, device=device)
        for j in range(m):
            mat_rel_singleton[j] = relation_matrix_torch(self.data[j], p=self.ps[j])
        FE_singleton = -torch.mean(torch.log2(mat_rel_singleton.mean(dim=1)), dim=1)
        # print('Entropy of attributes:',FE_singleton)

        fuzzy_mutual_info = []
        for i,j in its.combinations(range(m), 2):
            FE_ij = -torch.log2(torch.minimum(mat_rel_singleton[i], mat_rel_singleton[j]).mean(0)).mean(0)
            fmi = (FE_singleton[i] + FE_singleton[j] - FE_ij) / torch.sqrt(FE_singleton[i] * FE_singleton[j])
            # print(f"FE(X): {FE_singleton[i]}, FE(Y): {FE_singleton[j]}, FE(X,Y):{FE_ij}, NFMI:{fmi}")
            fuzzy_mutual_info.append(fmi.cpu())
        return np.array(fuzzy_mutual_info)


    def NFMI_v2(self):
        m, n, _ = self.data.shape
        FE_singleton = torch.zeros(m, dtype=torch.float32, device=device)
        fuzzy_mutual_info = []

        for i in range(m):
            mat_rel_singleton_i = relation_matrix_torch(self.data[i], p=self.ps[i])
            FE_singleton[i] = -torch.mean(torch.log2(mat_rel_singleton_i.mean(dim=1)))

        for i in range(m-1):
            mat_rel_singleton_i = relation_matrix_torch(self.data[i], p=self.ps[i])
            for j in range(i+1, m):
                # print(i,j)
                mat_rel_singleton_j = relation_matrix_torch(self.data[j], p=self.ps[j])
                FE_ij = -torch.log2(torch.minimum(mat_rel_singleton_i, mat_rel_singleton_j).mean(0)).mean(0)
                fmi = (FE_singleton[i] + FE_singleton[j] - FE_ij) / torch.sqrt(FE_singleton[i] * FE_singleton[j])
                # print(f"FE(X): {FE_singleton[i]}, FE(Y): {FE_singleton[j]}, FE(X,Y):{FE_ij}, NFMI:{fmi}")
                fuzzy_mutual_info.append(fmi.cpu())
        return np.array(fuzzy_mutual_info)


    def attribute_partition(self, tau=0.8):
        m, n, _ = self.data.shape
        subgroups = np.arange(m).reshape(-1, 1).tolist()
        idx = 0
        for left, right in its.combinations(np.arange(m), 2):
            if self.fmi[idx] > tau:
                subgroups = merge_set(subgroups, left, right)
            idx += 1
        print('\t\t\tAttribute partition:{}'.format(subgroups)[:50],'...')

        self.n_partitions = len(subgroups)
        if self.attr_partition is not None:
            if self.attr_partition == subgroups:
                return True

        self.attr_partition = subgroups
        return False


    def make_fuzzy_relative_entropy(self):
        m, n, _ = self.data.shape
        diag_idx = np.arange(n)
        self.cardinality_mean = torch.zeros((self.n_partitions, n), dtype=torch.float32, device=device)
        FE_B_x = torch.zeros((n, self.n_partitions), dtype=torch.float32, device=device)
        for j, bin in enumerate(self.attr_partition):
            if len(bin) == 1:
                bin = bin[0]
                dist_rel_mat = torch.cdist(self.data[bin], self.data[bin], p=1).float()
                if self.nominals[bin]:
                    dist_rel_mat.gt_(1e-5)
                else:
                    k = int(self.p * n * n)  # k-th minimum value = p-th percentile
                    delta = dist_rel_mat.view(-1).kthvalue(k).values
                    dist_rel_mat.masked_fill_(dist_rel_mat > delta, 1)
            else:
                dist_rel_mat = torch.zeros(n, n, dtype=torch.float32).to(device)
                for idx_, bin_ in enumerate(bin):
                    data_ = self.data[bin_]
                    temp_i = torch.cdist(data_, data_, p=1).float()
                    if self.nominals[bin_]:
                        temp_i.gt_(1e-5)
                    dist_rel_mat += temp_i
                dist_rel_mat /= len(bin)
                k = int(self.p * n * n)  # k-th minimum value = p-th percentile
                delta = dist_rel_mat.view(-1).kthvalue(k).values
                dist_rel_mat.masked_fill_(dist_rel_mat > delta, 1)

            dist_rel_mat.mul_(-1).add_(1)
            self.cardinality_mean[j] = dist_rel_mat.mean(dim=1)

            relation_matrix_mean_x = (torch.tile(dist_rel_mat.sum(dim=0), (n,1)) - dist_rel_mat) / (n - 1)
            relation_matrix_mean_x[diag_idx, diag_idx] = 1 # Equals to remove the value of each x as torch.log2(1) = 0
            FE_B_x[:, j] = -torch.sum(torch.log2(relation_matrix_mean_x), dim=1) / (n - 1)
            del relation_matrix_mean_x

        FE_B = -torch.mean(torch.log2(self.cardinality_mean), dim=1)
        self.FRE = (FE_B_x + 1e-6) / (FE_B + 1e-6) + 1/n


    def predict_score(self):
        self.make_fuzzy_relative_entropy()
        OD = 1 - (torch.sqrt(self.cardinality_mean) * self.FRE.T).mean(dim=0)
        return OD.cpu()
