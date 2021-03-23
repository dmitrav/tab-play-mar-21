
import pandas

def binarize_categorical_features(train, test):

    y_train = train['target']
    train = train.drop(['target'], axis=1)

    all_data = pandas.concat([train, test], axis=0)

    for i in range(19):
        ith_name = 'cat{}'.format(i)
        ith_features = pandas.get_dummies(all_data[ith_name], prefix=ith_name)

        all_data = pandas.concat([all_data, ith_features], axis=1)
        all_data = all_data.drop([ith_name], axis=1)

    train = all_data.iloc[:y_train.shape[0], :]
    train['target'] = y_train
    test = all_data.iloc[y_train.shape[0]:, :]

    return train, test


if __name__ == "__main__":
    pass