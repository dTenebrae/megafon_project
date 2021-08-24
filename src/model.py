import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from lightgbm import LGBMClassifier

from imblearn.over_sampling import RandomOverSampler
from collections import Counter

import warnings
warnings.simplefilter("ignore")

import joblib

import os
import sys

SEED = 42

green_list = [
        'id',
        'vas_id',
        'buy_time',
        '0',
        '1',
        '7',
        '9',
        '13',
        '19',
        '20',
        '25',
        '28',
        '36',
        '37',
        '38',
        '39',
        '43',
        '47',
        '48',
        '49',
        '50',
        '54',
        # '55',
        '56',
        '58',
        '59',
        '60',
        '61',
        '62',
        '64',
        '77',
        '103',
        '106',
        '110',
        '111',
        '114',
        '115',
        '126',
        '127',
        '128',
        '129',
        '130',
        '131',
        '133',
        '134',
        '135',
        '136',
        '143',
        '145',
        '148',
        '150',
        '159',
        '160',
        '164',
        '165',
        '191',
        '192',
        '193',
        '205',
        '207',
        '208',
        '209',
        '210',
        '211',
        '212',
        '213',
        '219',
        '222',
        '224',
        '226',
        '229',
        '230',
        '236',
        '237',
        '238',
        '239',
        '241',
        '242',
        '243',
        '245',
        '246',
        '247',
        '248',
        '250'
    ]

def eval_model(model, X, y):
    """
    Fitting given model and returns it. Also print some statistics
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
    model.fit(X_train, y_train)
    print("=" * 25)
    print(f"f1 score: {f1_score(model.predict(X_test), y_test, average='macro'):.3f}")
    print("=" * 25)
    return model

if __name__ == "__main__":
    cwd = os.getcwd()
    TEST = os.path.join(cwd, 'data/data_test.csv')
    MERGED_TRAIN = os.path.join(cwd, 'data/merged_train.pkl')
    MERGED_TEST = os.path.join(cwd, 'data/merged_test.pkl')
    MODEL_PATH = os.path.join(cwd, 'model/lgb_model.pkl')

    print("Open datasets")
    df_train = pd.read_pickle(MERGED_TRAIN)
    df_test = pd.read_pickle(MERGED_TEST)
    result_df = pd.read_csv(TEST)

    df_train['buy_time'] = df_train['buy_time'].astype('category')
    df_test['buy_time'] = df_test['buy_time'].astype('category')
    cat_feats = df_train.select_dtypes(include='category').columns.tolist()

    X = df_train.drop(['target'], axis=1)
    y = df_train['target'].astype(int)

    print("Drop some columns")
    X = X[green_list]
    df_test = df_test[X.columns.tolist()]


    print("Oversampling data")
    ros = RandomOverSampler(random_state=SEED)
    X_ros, y_ros = ros.fit_resample(X, y)
    print("=" * 25)
    print('Original dataset shape', Counter(y))
    print('Resample dataset shape', Counter(y_ros))
    print("=" * 25)


    print("Training model")
    lgb_model = LGBMClassifier(objective='binary',
                          max_depth=8,
                          n_estimators=300,
                          learning_rate=0.05,
                          categorical_column=cat_feats)
    lgb_model = eval_model(lgb_model, X_ros, y_ros)
    print("Saving model")
    joblib.dump(lgb_model, MODEL_PATH)

    print("Predicting test dataframe")
    prediction = lgb_model.predict_proba(df_test)[:, 1]
    result_df['target'] = prediction
    result_df.to_csv('answers_test.csv', index=False)

