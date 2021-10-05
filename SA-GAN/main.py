import sys
import os
import numpy as np
from pathlib import Path

import torch
from SA_GAN.data_utils import *
from SA_GAN.options import Options
from SA_GAN.model import *
from torch.utils.data import DataLoader


opt = Options().parse()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#print(device)

def set_seed(seed_value):
    if seed_value == -1:
        return

    import random

    random.seed(seed_value)

    torch.manual_seed(seed_value)

    torch.cuda.manual_seed(seed_value)

    torch.cuda.manual_seed_all(seed_value)

    np.random.seed(seed_value)

    torch.backends.cudnn.deterministic = True

seed = 12345

set_seed(seed)

batchsize = 50

opt.train_batchsize = batchsize
opt.val_batchsize = batchsize
opt.test_batchsize = batchsize

opt.niter = 20

kernel_size = 32

opt.ngf = kernel_size
opt.ndf = kernel_size

def anomaly_detection():

    #print('-------------------' + str(opt.delta) + '-----------------------')
    #print('-------------------' + str(opt.ratio) + '-----------------------')

    if not Path("./results").exists():
        Path("./results").mkdir()
    with open('./results/' + str(opt.dataset) + '_update' + str(opt.update_method) + '_' + str(opt.sample_method) + '_ratio=' + str(
            opt.ratio) + '_delta=' + str(opt.delta) + '_cnn.txt', 'w') as f:

        f.write('channel' + '\t' + 'dataset' + '\t' + 'f1' + '\t' + 'pre' + '\t' + 'rec' + '\t' +
                'tp' + '\t' + 'tn' + '\t' + 'fp' + '\t' + 'fn' + '\t' + 'train_time' + '\t' + 'epoch_time' +
                '\t' + 'test_time' + '\n')
        total_tp = 0.0
        total_tn = 0.0
        total_fp = 0.0
        total_fn = 0.0
        total_train_time = 0.0
        total_test_time = 0.0
        total_epoch_time = 0.0
        if opt.dataset == 'SMAP':
            opt.dim = 25
            opt.w_lat = 1
            opt.w_rec = 0.1
        elif opt.dataset == 'MSL':
            opt.dim = 55
            opt.w_lat = 1
            opt.w_rec = 1
        elif opt.dataset == 'SMD':
            opt.dim = 38
            opt.w_lat = 1
            opt.w_rec = 1
        elif opt.dataset == 'SWAT':
            opt.dim = 51
            opt.w_lat = 0.01
            opt.w_rec = 0.01
            opt.step = 10
        else:
            raise ValueError("Unrecognize dataset.")

        path_train = os.path.join(os.getcwd(), "datasets", "train", opt.dataset)
        files = os.listdir(path_train)
        file_number = 0
        for file in files:
            file_number += 1
            opt.filename = file
            seed = 12345
            set_seed(seed)
            data_name = opt.dataset + '/' + str(file)
            #print('file=', data_name)
            if opt.sample_method == 'seq':
                samples_train_data, samples_val_data = read_train_data_seq_based(opt.window_size, file=data_name,
                                                                   step=opt.step, delta=opt.delta, ratio=opt.ratio)
            elif opt.sample_method == 'win':
                samples_train_data, samples_val_data = read_train_data_win_based(opt.window_size, file=data_name,
                                                                   step=opt.step, delta=opt.delta, ratio=opt.ratio)
            elif opt.sample_method == 'sw':
                samples_train_data, samples_val_data = read_train_data_seq_win_based(opt.window_size, file=data_name,
                                                                             step=opt.step, delta=opt.delta,
                                                                             ratio=opt.ratio)
            else:
                raise ValueError('no such sampling method')
            train_data = DataLoader(dataset=samples_train_data, batch_size=opt.train_batchsize, shuffle=True)
            val_data = DataLoader(dataset=samples_val_data, batch_size=opt.val_batchsize, shuffle=True)

            samples_test_data, test_label = read_test_data(opt.window_size, file=data_name)

            test_data = DataLoader(dataset=samples_test_data, batch_size=opt.test_batchsize)

            alpha_rec = 0.001
            alpha_lat = 0.001

            model = SA(opt, train_data, val_data, test_data, test_label, device, alpha_rec, alpha_lat)

            train_time, epoch_time, flag = model.train()

            model.load()

            f1, pre, rec, tp, tn, fp, fn, test_time = model.eval_result(test_data, test_label,
                    "../task_assignment/data/time_series_data/SA-point_score/", opt.dataset, file)

            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_pre = total_tp / (total_tp + total_fp)
            total_rec = total_tp / (total_tp + total_fn)
            total_f1 = 2*total_pre*total_rec / (total_pre + total_rec)
            total_train_time += train_time
            total_test_time += test_time
            total_epoch_time += epoch_time
            if flag == 1:
                #print('update with method 1')
                pass
            elif flag == 2:
                #print('update with method 2')
                pass
            else:
                raise ValueError('no updating method')
            #print(str(opt.dataset) + '\t' + str(file) + '\tf1=' + str(f1) + '\tpre=' + str(pre) +
            #      '\trec=' + str(rec) + '\ttp=' + str(tp) + '\ttn=' + str(tn) + '\tfp=' + str(fp) +
            #      '\tfn=' + str(fn))

            #print('total results:' + '\tt_f1=' + str(total_f1) + '\tt_pre=' + str(total_pre) +
            #      '\tt_rec=' + str(total_rec) + '\tt_tp=' + str(total_tp) + '\tt_tn=' + str(total_tn) +
            #      '\tt_fp=' + str(total_fp) + '\tt_fn=' + str(total_fn) + '\tt_epoch_time='
            #      + str(total_epoch_time) + '\tt_test_time=' + str(total_test_time/file_number/opt.test_batchsize))

            f.write(str(opt.dataset) + '\t' + str(file) + '\t' + str(f1) + '\t' + str(pre) + '\t' +
                    str(rec) + '\t' + str(tp) + '\t' + str(tn) + '\t' + str(fp) + '\t' + str(fn) +
                    '\t' + str(train_time) + '\t' + str(epoch_time) + '\t' + str(test_time) + '\n')
        f.write('\n')
        f.write('total results' + '\t' + str(opt.dataset) + '\t' + str(total_f1) + '\t' + str(total_pre) + '\t' +
                str(total_rec) + '\t' + str(total_tp) + '\t' + str(total_tn) + '\t' + str(total_fp) + '\t' + str(total_fn) +
                '\t' + str(total_train_time) + '\t' + str(total_epoch_time) + '\t'
                + str(total_test_time/file_number/opt.test_batchsize) +'\n')


    #print('finished')


if __name__ == '__main__':

    anomaly_detection()
