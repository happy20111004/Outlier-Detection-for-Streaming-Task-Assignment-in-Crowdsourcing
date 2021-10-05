import numpy as np
from  torch.utils.data import DataLoader,TensorDataset
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import copy



def proprocess(df):

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError("Data must be 2-D array")

    if np.any(sum(np.isnan(df)) != 0):
        #print("Data contains nan. Will be repalced with 0")
        pass

        df = np.nan_to_num()

    df = MinMaxScaler().fit_transform(df)

    #print("Data is normalized [0,1]")

    return df


def read_train_data_seq_based(seq_length, file = '', step=1, valid_portition=0.3, delta=0.05, ratio=0.01):

    df1 = np.load('./datasets/train/' + file, allow_pickle=True)
    #print(df1.shape)

    #print('negative samples based on sequence')

    #print('ratio of negative samples:', ratio)
    (whole_len, whole_dim) = df1.shape


    values_y = proprocess(df1)


    df = np.load('./datasets/train/' + file, allow_pickle=True)
    #print(df.shape)

    (whole_len, whole_dim) = df.shape



    values = proprocess(df)

    length = int(whole_len * ratio)

    if length < 1:
        length = 1


    index = np.random.randint(0, whole_len, length)


    for i in range(whole_dim):
        low_val = min(values[:, i])
        high_val = max(values[:, i])
        delta_val = high_val - low_val

        a = np.random.uniform(
            low=low_val - delta * delta_val,
            high=high_val + delta * delta_val,
            size=length)

        for j in range(len(index)):
            values[index[j]][i] = a[j]


    n = int(len(values) * valid_portition)

    if n > seq_length:
        train, val = values[:-n], values[-n:]
        train_y, val_y = values_y[:-n], values_y[-n:]

        if step == 1:
            num_samples_train = (train.shape[0] - seq_length) + 1
            num_samples_val = (val.shape[0] - seq_length) + 1

        else:
            num_samples_train = (train.shape[0] - seq_length) // step
            num_samples_val = (val.shape[0] - seq_length) // step

        temp_train = np.empty([num_samples_train, seq_length, whole_dim])
        temp_train_y = np.empty([num_samples_train, seq_length, whole_dim])

        temp_val = np.empty([num_samples_val, seq_length, whole_dim])
        temp_val_y = np.empty([num_samples_val, seq_length, whole_dim])


        for i in range(num_samples_train):
            for j in range(train.shape[1]):
                temp_train[i, :, j] = train[(i*step):(i*step + seq_length), j]
                temp_train_y[i, :, j] = train_y[(i * step):(i * step + seq_length), j]


        for i in range(num_samples_val):
            for j in range(val_y.shape[1]):
                temp_val[i, :, j] = val_y[(i*step):(i*step + seq_length), j]
                temp_val_y[i, :, j] = val_y[(i * step):(i * step + seq_length), j]

        train_data_x = temp_train
        train_data_y = temp_train_y

        train_data = TensorDataset(torch.Tensor(train_data_x), torch.Tensor(train_data_y))

        val_data_x = temp_val
        val_data_y = temp_val_y

        val_data = TensorDataset(torch.Tensor(val_data_x), torch.Tensor(val_data_y))

    else:
        train = values
        train_y = values_y

        if step == 1:
            num_samples_train = (train.shape[0] - seq_length) + 1


        else:
            num_samples_train = (train.shape[0] - seq_length) // step


        temp_train = np.empty([num_samples_train, seq_length, whole_dim])
        temp_train_y = np.empty([num_samples_train, seq_length, whole_dim])


        for i in range(num_samples_train):
            for j in range(train.shape[1]):
                temp_train[i, :, j] = train[(i * step):(i * step + seq_length), j]
                temp_train_y[i, :, j] = train_y[(i * step):(i * step + seq_length), j]

        train_data_x = temp_train
        train_data_y = temp_train_y
        train_data = TensorDataset(torch.Tensor(train_data_x), torch.Tensor(train_data_y))

        val_data_x = train_data_y
        val_data_y = train_data_y
        val_data = TensorDataset(torch.Tensor(val_data_x), torch.Tensor(val_data_y))

    return train_data, val_data


def neg_samples(data, index, delta, seq_length, ratio):

    data1 = copy.deepcopy(data)

    if int(seq_length * ratio) < 1:
        length = 1
    else:
        length = int(seq_length * ratio)

    low_val = min(data)
    high_val = max(data)
    delta_val = high_val - low_val

    a = np.random.uniform(
        low=low_val - delta * delta_val,
        high=high_val + delta * delta_val,
        size=length)

    for j in range(len(index)):
        data1[index[j]] = a[j]

    return data1


def neg_samples_seq_win(data, index, delta, seq_length, ratio):

    data1 = copy.deepcopy(data)

    if int(seq_length * ratio) < 1:
        length = 1  #
    else:
        length = int(seq_length * ratio)

    low_val = min(data)
    high_val = max(data)

    flag = np.random.randint(0,2)

    if flag == 0:  #

        a = np.random.uniform(
            low=low_val - delta,
            high=low_val,
            size=length)
    else:
        a = np.random.uniform(
            low=high_val,
            high=high_val + delta,
            size=length)

    for j in range(len(index)):
        data1[index[j]] = a[j]

    return data1

def read_train_data_win_based(seq_length, file = '', step=1, valid_portition=0.3, delta=0.05, ratio=0.01):

    #
    df1 = np.load('./datasets/train/' + file, allow_pickle=True)
    #print(df1.shape)

    #print('negative samples based on windows')


    values_y = proprocess(df1)
    #print('ratio of negative samples:', ratio)
    #
    df = np.load('./datasets/train/' + file, allow_pickle=True)
    #print(df.shape)

    (whole_len, whole_dim) = df.shape


    values = proprocess(df)

    #

    n = int(len(values) * valid_portition)

    if n > seq_length:#
        train, val = values[:-n], values[-n:]
        train_y, val_y = values_y[:-n], values_y[-n:]

        if step == 1:
            num_samples_train = (train.shape[0] - seq_length) + 1
            num_samples_val = (val.shape[0] - seq_length) + 1

        else:
            num_samples_train = (train.shape[0] - seq_length) // step
            num_samples_val = (val.shape[0] - seq_length) // step

        temp_train = np.empty([num_samples_train, seq_length, whole_dim])
        temp_train_y = np.empty([num_samples_train, seq_length, whole_dim])

        temp_val = np.empty([num_samples_val, seq_length, whole_dim])
        temp_val_y = np.empty([num_samples_val, seq_length, whole_dim])


        for i in range(num_samples_train):
            index = np.random.randint(0, seq_length, int(seq_length*ratio))
            for j in range(train.shape[1]):
                if max(train[(i * step):(i * step + seq_length), j]) > 1.1:
                    #print('val error')
                    pass
                temp_train[i, :, j] = neg_samples(train[(i*step):(i*step + seq_length), j], index, delta, seq_length, ratio)
                temp_train_y[i, :, j] = train_y[(i * step):(i * step + seq_length), j]


        for i in range(num_samples_val):
            for j in range(val_y.shape[1]):
                temp_val[i, :, j] = val_y[(i*step):(i*step + seq_length), j]
                temp_val_y[i, :, j] = val_y[(i * step):(i * step + seq_length), j]


        train_data_x = temp_train
        train_data_y = temp_train_y

        train_data = TensorDataset(torch.Tensor(train_data_x), torch.Tensor(train_data_y))

        val_data_x = temp_val
        val_data_y = temp_val_y

        val_data = TensorDataset(torch.Tensor(val_data_x), torch.Tensor(val_data_y))

    else:
        train = values
        train_y = values_y

        if step == 1:
            num_samples_train = (train.shape[0] - seq_length) + 1


        else:
            num_samples_train = (train.shape[0] - seq_length) // step


        temp_train = np.empty([num_samples_train, seq_length, whole_dim])
        temp_train_y = np.empty([num_samples_train, seq_length, whole_dim])


        for i in range(num_samples_train):
            for j in range(train.shape[1]):
                temp_train[i, :, j] = train[(i * step):(i * step + seq_length), j]
                temp_train_y[i, :, j] = train_y[(i * step):(i * step + seq_length), j]

        train_data_x = temp_train
        train_data_y = temp_train_y
        train_data = TensorDataset(torch.Tensor(train_data_x), torch.Tensor(train_data_y))

        val_data_x = train_data_y
        val_data_y = train_data_y
        val_data = TensorDataset(torch.Tensor(val_data_x), torch.Tensor(val_data_y))

    return train_data, val_data



def read_train_data_seq_win_based(seq_length, file = '', step=1, valid_portition=0.3, delta=0.05, ratio=0.01):

    #
    df1 = np.load('./datasets/train/' + file, allow_pickle=True).astype(np.float)
    #print(df1.shape)

    #print('negative samples based on seq-win')


    values_y = proprocess(df1)
    #print('ratio of negative samples:', ratio)
    #
    df = np.load('./datasets/train/' + file, allow_pickle=True).astype(np.float)
    #print(df.shape)

    (whole_len, whole_dim) = df.shape



    values = df

    length = int(whole_len * ratio)
    if length < 1:
        length = 1

    #
    index = np.random.randint(0, whole_len, length)

    for i in range(whole_dim):

        low_val = min(values[:, i])
        high_val = max(values[:, i])


        flag  = np.random.randint(0,2)

        if flag == 0:#

            a = np.random.uniform(
                low=low_val-delta,
                high=low_val,
                size=length)
        else:
            a = np.random.uniform(
                low=high_val,
                high=high_val + delta,
                size=length)

        for j in range(len(index)):
            values[index[j]][i] = a[j]


    n = int(len(values) * valid_portition)


    if n > seq_length:#
        train, val = values[:-n], values[-n:]
        train_y, val_y = values_y[:-n], values_y[-n:]

        if step == 1:
            num_samples_train = (train.shape[0] - seq_length) + 1
            num_samples_val = (val.shape[0] - seq_length) + 1

        else:
            num_samples_train = (train.shape[0] - seq_length) // step
            num_samples_val = (val.shape[0] - seq_length) // step



        temp_train = np.empty([num_samples_train, seq_length, whole_dim])
        temp_train_y = np.empty([num_samples_train, seq_length, whole_dim])

        temp_val = np.empty([num_samples_val, seq_length, whole_dim])
        temp_val_y = np.empty([num_samples_val, seq_length, whole_dim])

        for i in range(num_samples_train):
            index = np.random.randint(0, seq_length, int(seq_length * ratio))
            for j in range(train.shape[1]):
                temp_train[i, :, j] = neg_samples_seq_win(train[(i * step):(i * step + seq_length), j],
                                                                      index, delta, seq_length, ratio)
                temp_train_y[i, :, j] = train_y[(i * step):(i * step + seq_length), j]

        seq_win = np.concatenate(temp_train)

        seq_win = proprocess(seq_win)


        for i in range(num_samples_train):
            for j in range(train.shape[1]):
                temp_train[i, :, j] = seq_win[(i * step):(i * step + seq_length), j]


        for i in range(num_samples_val):
            for j in range(val_y.shape[1]):
                temp_val[i, :, j] = val_y[(i*step):(i*step + seq_length), j]
                temp_val_y[i, :, j] = val_y[(i * step):(i * step + seq_length), j]


        train_data_x = temp_train
        train_data_y = temp_train_y

        train_data = TensorDataset(torch.Tensor(train_data_x), torch.Tensor(train_data_y))

        val_data_x = temp_val
        val_data_y = temp_val_y

        val_data = TensorDataset(torch.Tensor(val_data_x), torch.Tensor(val_data_y))

    else:
        train = values
        train_y = values_y

        if step == 1:
            num_samples_train = (train.shape[0] - seq_length) + 1


        else:
            num_samples_train = (train.shape[0] - seq_length) // step


        temp_train = np.empty([num_samples_train, seq_length, whole_dim])
        temp_train_y = np.empty([num_samples_train, seq_length, whole_dim])


        for i in range(num_samples_train):
            for j in range(train.shape[1]):
                temp_train[i, :, j] = train[(i * step):(i * step + seq_length), j]
                temp_train_y[i, :, j] = train_y[(i * step):(i * step + seq_length), j]

        train_data_x = temp_train
        train_data_y = temp_train_y
        train_data = TensorDataset(torch.Tensor(train_data_x), torch.Tensor(train_data_y))

        val_data_x = train_data_y
        val_data_y = train_data_y
        val_data = TensorDataset(torch.Tensor(val_data_x), torch.Tensor(val_data_y))

    return train_data, val_data


def read_test_data(seq_length, file = ''):

    df = np.load('./datasets/test/' + file, allow_pickle=True)
    label = np.load('./datasets/test_label/' + file, allow_pickle=True).astype(np.float)
    #print(df.shape, label.shape)

    (whole_len, whole_dim) = df.shape


    test = proprocess(df)



    num_samples_test = (test.shape[0] - seq_length) + 1

    temp_test = np.empty([num_samples_test, seq_length, whole_dim])


    for i in range(num_samples_test):
        for j in range(test.shape[1]):
                temp_test[i, :, j] = test[(i):(i + seq_length), j]


    test_data = temp_test

    return test_data, label


