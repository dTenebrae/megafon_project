import pandas as pd
import numpy as np
from tqdm.auto import tqdm # progress bar for column transform
import os
import sys
import warnings
warnings.simplefilter("ignore")


def col_transform(df: pd.DataFrame):
    """
    Transform or drop columns based on number of unique elements in it
    """
    for col in tqdm(df.columns):
        col_nunique = df[col].nunique()
        if col_nunique == 1:
            df.drop(col, axis=1, inplace=True)
            continue
        elif col_nunique == 2:
            first_value = df[col].iloc[0]
            df[col] = np.where(df[col] == first_value, 1, 0)
            df[col] = df[col].astype(np.uint8)
            continue

        diff = df[col].fillna(0).astype(int).sum() - df[col].fillna(0).sum()
        if diff == 0:
            df[col] = df[col].astype(np.int32)
    # TODO Check if buy_time is in columns
    df.sort_values(by=['buy_time'], inplace=True)


def drop_correlated(df: pd.DataFrame, threshold=0.9):
    """
    Drop column with high correlation (above threshold)
    """
    cols_before = df.shape[1]
    corr = df.corr().abs()
    mask = np.ones(corr.shape)
    triangular_mask = np.triu(mask, k=1).astype(np.bool)
    upper_triu = corr.where(triangular_mask)
    to_drop = [column for column in tqdm(upper_triu.columns) if any(upper_triu[column] > threshold)]
    df.drop(df[to_drop], axis=1, inplace=True)
    print(f"Columns before: {cols_before}\nColumns after: {df.shape[1]}")


def constant_columns(df: pd.DataFrame, thr=0.99) -> list():
    """
    Returns list of columns with one value percentage more than threshold
    """
    almost_const = []
    for col in tqdm(df.columns[4:]):
        perc = df[col].value_counts(normalize=True).iloc[0]
        if perc >= thr:
            almost_const.append(col)
    return almost_const

def categorical_columns(df: pd.DataFrame, thr=20) -> list():
    """
    Returns list of columns with number of unique values below threshold
    """
    result = []
    for col in tqdm(df.columns):
        if df[col].nunique() <= thr:
            result.append(col)
    return result

def process_feat(df: pd.DataFrame):
    const_feats = constant_columns(df, thr=0.9)
    df.drop(const_feats, axis=1, inplace=True)
    drop_correlated(df, threshold=0.8)

def process_merged(df: pd.DataFrame) -> pd.DataFrame:
    df_merged = pd.merge_asof(df, df_feat, on='buy_time', by='id', direction='nearest')
    df_merged['buy_time'] = df_merged.apply(lambda x: pd.Timestamp.fromtimestamp(x['buy_time']), axis=1)
    df_merged['buy_time'] = df_merged['buy_time'].dt.weekofyear

    cat_feats = categorical_columns(df_merged, thr=50)
    df_merged[cat_feats] = df_merged[cat_feats].astype('category')

    return df_merged



if __name__ == "__main__":
    cwd = os.getcwd()

    TRAIN = os.path.join(cwd, 'data/data_train.csv')
    TEST = os.path.join(cwd, 'data/data_test.csv')
    FEAT = os.path.join(cwd, 'data/features.csv')
    PROC_FEAT = os.path.join(cwd, 'data/processed_features.pkl')
    PROC_TRAIN = os.path.join(cwd, 'data/processed_train.pkl')
    PROC_TEST = os.path.join(cwd, 'data/processed_test.pkl')
    MERGED_TRAIN = os.path.join(cwd, 'data/merged_train.pkl')
    MERGED_TEST = os.path.join(cwd, 'data/merged_test.pkl')

    try:
        print("Opening train file...")
        df_train = pd.read_csv(TRAIN,
                               dtype={
                                   'id': np.int32,
                                   'buy_time': np.int32
                               }).drop('Unnamed: 0', axis=1)

        print("Opening test file...")
        df_test = pd.read_csv(TEST,
                              dtype={
                                  'id': np.int32,
                                  'buy_time': np.int32
                              }).drop('Unnamed: 0', axis=1)
        # create list of users id, which exists in train and test datasets
        print("Collect all users id")
        users_id = list(set(df_train['id'].tolist()).union(set(df_test['id'].tolist())))
        df_train.sort_values(by=['buy_time'], inplace=True)
        df_test.sort_values(by=['buy_time'], inplace=True)
        print("Saving to file")
        df_train.to_pickle(PROC_TRAIN)
        df_test.to_pickle(PROC_TEST)
        # del df_train
        # del df_test
    except FileNotFoundError:
        print("File not found. Place data_train.csv and data_test.csv in data folder of a project")
        sys.exit(1)

    try:
        print("Opening feature file...")
        df_feat = pd.read_csv(FEAT,
                              sep='\t',
                              dtype={str(i): np.float32 for i in range(253)}).drop('Unnamed: 0', axis=1)
        # remove all unnecessary data
        print("Removing nonexistent id")
        df_feat = df_feat[df_feat['id'].isin(users_id)]
        # transform columns in a dataset
        print("Transforming columns...")
        col_transform(df_feat)
        # save our trimmed features dataset as pickle file
        print("Saving to file")
        df_feat.to_pickle(PROC_FEAT)
        # del df_feat
    except FileNotFoundError:
        print("File not found. Place features.csv in data folder of a project")
        sys.exit(1)

    print("Processing feat dataset")
    process_feat(df_feat)

    print("Processing train dataset")
    df_train_merged = process_merged(df_train)
    print("Processing test dataset")
    df_test_merged = process_merged(df_test)

    print("Saving datasets")
    df_train_merged.to_pickle(MERGED_TRAIN)
    df_test_merged.to_pickle(MERGED_TEST)

