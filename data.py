import pandas as pd
import numpy as np
import logging
import random


def load_data(path):
    data = pd.read_csv(path, delimiter=',')
    """
    Year,Month,Day,Hour,Value,Value1,Value2,Value3,dayOfWeek,isWorkday,isHoliday,Season,
    Tem,RH,Precipitation,File,value_oneweek_before,value_oneday_before,value_onedayavg_before
    """
    # names = ['Month', 'Day', 'Hour', 'dayOfWeek', 'isWorkday', 'isHoliday', 'Season', 'Tem', 'RH',
    #          'value_oneweek_before', 'value_oneday_before', 'value_onedayavg_before', 'Value']

    # names = ['dayOfWeek', 'isWorkday', 'isHoliday', 'Season', 'Tem', 'RH',
    #          'value_oneweek_before', 'value_oneday_before', 'value_onedayavg_before', 'Value']
    #
    # data = df[names].values

    index_zero_value = []
    for i in range(data.shape[0]):
        if data['Value'][i] == 0:
            index_zero_value.append(i)
    df = data.loc[:]
    for i in index_zero_value:
        df.loc[i, 'Value'] = None
    df = df.dropna()
    # end
    max_value = np.max(df['Value'])
    min_value = np.min(df['Value'])
    dfy = pd.DataFrame({'Value': (df['Value'] - min_value) / (max_value - min_value)})
    dfX = pd.DataFrame({'dayOfWeek': df['dayOfWeek'],
                        'isWorkday': df['isWorkday'], 'isHoliday': df['isHoliday'],
                        'Season': df['Season'],
                        'Tem': (df['Tem'] - np.mean(df['Tem'])) / (np.max(df['Tem']) - np.min(df['Tem'])),
                        'RH': (df['RH'] - np.mean(df['RH'])) / (np.max(df['RH']) - np.min(df['RH']))})
    df_X = np.array(dfX)
    df_y = np.array(dfy)
    data_ = np.concatenate((df_X, df_y), axis=1)
    return data_, max_value, min_value


def get_train_data(data, shuffle=False, input_size=9, batch_size=60, time_step=15, train_begin=0, train_end=2000):
    train_data = data[train_begin:train_end]

    if shuffle:
        random.shuffle(data)

    # 标准化
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)

    # normalized_train_data = (train_data - mean) / std
    normalized_train_data = train_data

    train_x, train_y = [], []  # 训练集
    for i in range(len(normalized_train_data) - time_step):
        if len(train_x) == batch_size:
            yield train_x, train_y
            train_x, train_y = [], []
        x = normalized_train_data[i:i + time_step, :input_size]
        y = normalized_train_data[i:i + time_step, input_size, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())


def get_test_data(data, input_size=6, time_step=15, test_begin=2000, test_end=2500):
    test_data = data[test_begin:test_end]

    mean = np.mean(test_data, axis=0)
    std = np.std(test_data, axis=0)

    # normalized_test_data = (test_data - mean) / std
    normalized_test_data = test_data

    size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample

    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :input_size]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, input_size]
        test_x.append(x.tolist())
        test_y.extend(y)

    return test_x, test_y


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


if __name__ == '__main__':
    data, max_value, min_value = load_data('./data_path/HourLoadSet.csv')
    # batches = get_train_data(data)
    test_x, test_y = get_test_data(data=data, test_begin=15000, test_end=17000)
    test_y = np.array(test_y)

    test_y_ = test_y * (max_value - min_value) + min_value

    print(max_value, min_value)
    for i, j in zip(test_y_, test_y):
        print(i, j)
