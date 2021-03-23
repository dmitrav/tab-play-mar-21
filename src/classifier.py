
import numpy, pandas
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score


from src import preprocessor

if __name__ == "__main__":

    train_path = "../data/train.csv"
    test_path = "../data/test.csv"

    train = pandas.read_csv(train_path)
    test = pandas.read_csv(test_path)

    # train, test = preprocessor.binarize_categorical_features(train, test)

    X, y = train.drop(['target'], axis=1), train['target']

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.75, random_state=42)

    model = CatBoostClassifier(
        custom_loss=['Accuracy'],
        random_seed=42,
        logging_level='Silent'
    )

    categorical_features_indices = numpy.where(X.dtypes != numpy.float)[0]

    model.fit(
        X_train, y_train,
        cat_features=categorical_features_indices,
        eval_set=(X_val, y_val),
        #     logging_level='Verbose',  # you can uncomment this for text output
        plot=True
    )

    cv_params = model.get_params()
    cv_params.update({
        'loss_function': 'Logloss'
    })
    cv_data = cv(
        Pool(X, y, cat_features=categorical_features_indices),
        cv_params,
        plot=True
    )

    print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
        numpy.max(cv_data['test-Accuracy-mean']),
        cv_data['test-Accuracy-std'][numpy.argmax(cv_data['test-Accuracy-mean'])],
        numpy.argmax(cv_data['test-Accuracy-mean'])
    ))

    print('Precise validation accuracy score: {}'.format(numpy.max(cv_data['test-Accuracy-mean'])))
