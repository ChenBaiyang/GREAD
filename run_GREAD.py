import os, time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model import GREAD

result_file_name = 'results_GREAD.csv'
open(result_file_name, 'w').write('dataset,model,threshold,lamb,auc,pr\n')

data_dir = 'data/'
dataset_list = os.listdir(data_dir)

for dataset in dataset_list:
    d = np.load(data_dir+dataset)
    data = d['X']
    n, m = data.shape
    valids = np.std(data, axis=0) > 1e-6
    data = data[:, valids]
    label = d['y']
    nominals = d['nominals'][valids]
    print("{}\t\t shape:{}\t# Outlier:{}\t# Nominals:{}".format(dataset[:-4], (n, m), label.sum(),nominals.sum()))

    paras = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    count1, count2 = 0, 0
    for lamb in paras:
        t0 = time.time()
        model = GREAD(data, nominals, lamb=lamb)
        print('Time={:.3f}'.format(time.time()-t0))
        for threshold in paras[::-1]:
            t1 = time.time()
            repeated = model.attribute_partition(threshold=threshold)
            if repeated:
                print('\t\t\tPartition unchanged...')
                # Use previous results
            else:
                out_scores = model.predict_score()
                auc = roc_auc_score(label, out_scores)
                pr = average_precision_score(y_true=label, y_score=out_scores, pos_label=1)
                count1 +=1

            print('\t\tThreshold={}\tAUC={:.4f}\tPR={:.4f}\tTime={:.3f}'.format(threshold, auc, pr, time.time()-t1))
            scores = [dataset[:-4], 'GREAD', str(threshold), str(lamb), str(auc)[:8], str(pr)[:8]]
            open(result_file_name, 'a').write(','.join(scores) + '\n')
            count2 += 1

        print(f'\t\tCounted:{count1}/{count2}')
        del model
    break
