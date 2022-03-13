from sklearn.model_selection import train_test_split
import errno
import os
import pandas as pd
import numpy as np



def load_MLQE_data(language, label, prep_for_training=False):
    # This method expects a folder per language

    assert language in ['EN-DE', 'EN-ZH', 'RO-EN', 'ET-EN', 'SI-EN', 'NE-EN', 'RU-EN']
    assert label in ['HTER', 'DA']

    # Set the path (Google Drive)
    path = 'DATA/MLQE-PE/' + language
    if not os.path.isdir(path):
        raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    # Import direct assessment data (train, dev & test)
    df_train = pd.read_csv(path + '/DA/da_train.tsv', engine="python", delimiter="\\t")
    df_dev = pd.read_csv(path + '/DA/da_dev.tsv', engine="python", delimiter="\\t")
    df_test = pd.read_csv(path + '/DA/da_test.tsv', engine="python", delimiter="\\t")

    # Import HTER scores (train, dev & test)
    df_train_HTER = pd.read_csv(path + '/HTER/train.hter', engine="python", delimiter="\\t", names=['HTER'])
    df_dev_HTER = pd.read_csv(path + '/HTER/dev.hter', engine="python", delimiter="\\t", names=['HTER'])
    df_test_HTER = pd.read_csv(path + '/HTER/test.hter', engine="python", delimiter="\\t", names=['HTER'])

    df_train = pd.concat((df_train, df_train_HTER), axis=1)
    df_dev = pd.concat((df_dev, df_dev_HTER), axis=1)
    df_test = pd.concat((df_test, df_test_HTER), axis=1)
    if 'index' not in list(df_test.columns):
        df_test['index'] = range(len(df_test))

    # Load the fluent but inadequate labels for the EN-DE test set
    if language == 'EN-DE':
        df_fluent_inadequate = pd.read_csv(path + '/hand-labelled/hand-labelled-test.csv')
        df_test = pd.concat((df_test, df_fluent_inadequate), axis=1)
        df_focal_weights = pd.read_csv(path+'/'+label+'/focal_weights.csv')
        df_train = pd.concat((df_train, df_focal_weights), axis=1)
    if language == 'EN-ZH':
        df_focal_weights = pd.read_csv(path+'/'+label+'/focal_weights.csv')
        df_train = pd.concat((df_train, df_focal_weights), axis=1)

    if prep_for_training:
        if label == 'HTER':
            train = df_train[['original', 'translation', 'HTER']]
            if language == 'EN-DE' or language == 'EN-ZH':
                train = df_train[['original', 'translation', 'HTER', 'focal_weights']]
            dev = df_dev[['original', 'translation', 'HTER']]
            test = df_test[['index', 'original', 'translation']]
            train = train.rename(columns={'original': 'text_a', 'translation': 'text_b', 'HTER': 'labels'})
            dev = dev.rename(columns={'original': 'text_a', 'translation': 'text_b', 'HTER': 'labels'})
            test = test.rename(columns={'original': 'text_a', 'translation': 'text_b'})
        if label == 'DA':
            train = df_train[['original', 'translation', 'z_mean']]
            if language == 'EN-DE' or language == 'EN-ZH':
                train = df_train[['original', 'translation', 'z_mean', 'focal_weights']]
            dev = df_dev[['original', 'translation', 'z_mean']]
            test = df_test[['index', 'original', 'translation']]
            train = train.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'})
            dev = dev.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'})
            test = test.rename(columns={'original': 'text_a', 'translation': 'text_b'})

        return train, dev, test
    else:
        return df_train, df_dev, df_test


def load_WikiMatrix_data(file, language, augmentation_type="shuffle", label="DA", prep_for_training = False):

    # High-quality WikiMatrix data is not available for every language pair.
    assert language in ['EN-DE', 'EN-ZH']

    path = 'DATA/WikiMatrix/' + language + '/'
    if not os.path.isfile(path+file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path+file)

    df_total = pd.read_csv(path+file, engine = "python", delimiter = "\\t")
    df_total["shuffled_source"] = df_total["source"].sample(frac=1, random_state=1).values
    df_train, df_test = train_test_split(df_total, test_size=1000, random_state=1)
    df_train, df_dev = train_test_split(df_train, test_size=1000, random_state=1)


    if label=="DA":
        if augmentation_type == "shuffle":
            df_train_good = df_train[['source', 'target']].copy()
            df_train_good['labels'] = np.ones(len(df_train)).astype(int)
            df_train_good = df_train_good[:3500]
            df_train_bad = df_train[['shuffled_source', 'target']].copy()
            df_train_bad['labels'] = np.zeros(len(df_train)).astype(int)
            df_train_bad = df_train_bad.rename(columns={"shuffled_source": "source"})
            df_train_bad = df_train_bad[3500:]
            df_train = pd.concat((df_train_good, df_train_bad), ignore_index=True)
            df_train = df_train.sample(frac=1, random_state=1)

            df_dev_good = df_dev[['source', 'target']].copy()
            df_dev_good['labels'] = np.ones(len(df_dev)).astype(int)
            df_dev_good = df_dev_good[:500]
            df_dev_bad = df_dev[['shuffled_source', 'target']].copy()
            df_dev_bad = df_dev_bad.rename(columns={"shuffled_source": "source"})
            df_dev_bad['labels'] = np.zeros(len(df_dev)).astype(int)
            df_dev_bad = df_dev_bad[500:]
            df_dev = pd.concat((df_dev_good, df_dev_bad), ignore_index=True)
            df_dev = df_dev.sample(frac=1, random_state=1)

            df_test_good = df_test[['source', 'target']].copy()
            df_test_good['labels'] = np.ones(len(df_test)).astype(int)
            df_test_good = df_test_good[:500]
            df_test_bad = df_test[['shuffled_source', 'target']].copy()
            df_test_bad = df_test_bad.rename(columns={"shuffled_source": "source"})
            df_test_bad['labels'] = np.zeros(len(df_test)).astype(int)
            df_test_bad = df_test_bad[500:]
            df_test = pd.concat((df_test_good, df_test_bad), ignore_index=True)
            df_test = df_test.sample(frac=1, random_state=1)
            df_test['index'] = np.arange(0, len(df_test))

        else:
            df_train_good = df_train[['source', 'target']].copy()
            df_train_good['labels'] = np.ones(len(df_train)).astype(int)
            df_train_good = df_train_good[:3500]
            df_train_bad = df_train[['source', 'target_augmented']].copy()
            df_train_bad['labels'] = np.zeros(len(df_train)).astype(int)
            df_train_bad = df_train_bad.rename(columns={"target_augmented": "target"})
            df_train_bad = df_train_bad[3500:]
            df_train = pd.concat((df_train_good, df_train_bad), ignore_index=True)
            df_train = df_train.sample(frac=1, random_state=1)

            df_dev_good = df_dev[['source', 'target']].copy()
            df_dev_good['labels'] = np.ones(len(df_dev)).astype(int)
            df_dev_good = df_dev_good[:500]
            df_dev_bad = df_dev[['source', 'target_augmented']].copy()
            df_dev_bad = df_dev_bad.rename(columns={"target_augmented": "target"})
            df_dev_bad['labels'] = np.zeros(len(df_dev)).astype(int)
            df_dev_bad = df_dev_bad[500:]
            df_dev = pd.concat((df_dev_good, df_dev_bad), ignore_index=True)
            df_dev = df_dev.sample(frac=1, random_state=1)

            df_test_good = df_test[['source', 'target']].copy()
            df_test_good['labels'] = np.ones(len(df_test)).astype(int)
            df_test_good = df_test_good[:500]
            df_test_bad = df_test[['source', 'target_augmented']].copy()
            df_test_bad = df_test_bad.rename(columns={"target_augmented": "target"})
            df_test_bad['labels'] = np.zeros(len(df_test)).astype(int)
            df_test_bad = df_test_bad[500:]
            df_test = pd.concat((df_test_good, df_test_bad), ignore_index=True)
            df_test = df_test.sample(frac=1, random_state=1)
            df_test['index']=np.arange(0,len(df_test))
    else:
        if augmentation_type == "shuffle":
            df_train_good = df_train[['source', 'target']].copy()
            df_train_good['labels'] = np.zeros(len(df_train)).astype(int)
            df_train_good = df_train_good[:3500]
            df_train_bad = df_train[['shuffled_source', 'target']].copy()
            df_train_bad['labels'] = np.ones(len(df_train)).astype(int)
            df_train_bad = df_train_bad.rename(columns={"shuffled_source": "source"})
            df_train_bad = df_train_bad[3500:]
            df_train = pd.concat((df_train_good, df_train_bad), ignore_index=True)
            df_train = df_train.sample(frac=1, random_state=1)

            df_dev_good = df_dev[['source', 'target']].copy()
            df_dev_good['labels'] = np.zeros(len(df_dev)).astype(int)
            df_dev_good = df_dev_good[:500]
            df_dev_bad = df_dev[['shuffled_source', 'target']].copy()
            df_dev_bad = df_dev_bad.rename(columns={"shuffled_source": "source"})
            df_dev_bad['labels'] = np.ones(len(df_dev)).astype(int)
            df_dev_bad = df_dev_bad[500:]
            df_dev = pd.concat((df_dev_good, df_dev_bad), ignore_index=True)
            df_dev = df_dev.sample(frac=1, random_state=1)

            df_test_good = df_test[['source', 'target']].copy()
            df_test_good['labels'] = np.zeros(len(df_test)).astype(int)
            df_test_good = df_test_good[:500]
            df_test_bad = df_test[['shuffled_source', 'target']].copy()
            df_test_bad = df_test_bad.rename(columns={"shuffled_source": "source"})
            df_test_bad['labels'] = np.ones(len(df_test)).astype(int)
            df_test_bad = df_test_bad[500:]
            df_test = pd.concat((df_test_good, df_test_bad), ignore_index=True)
            df_test = df_test.sample(frac=1, random_state=1)
            df_test['index'] = np.arange(0, len(df_test))

        else:
            df_train_good = df_train[['source', 'target']].copy()
            df_train_good['labels'] = np.zeros(len(df_train)).astype(int)
            df_train_good = df_train_good[:3500]
            df_train_bad = df_train[['source', 'target_augmented']].copy()
            df_train_bad['labels'] = np.ones(len(df_train)).astype(int)
            df_train_bad = df_train_bad.rename(columns={"target_augmented": "target"})
            df_train_bad = df_train_bad[3500:]
            df_train = pd.concat((df_train_good, df_train_bad), ignore_index=True)
            df_train = df_train.sample(frac=1, random_state=1)

            df_dev_good = df_dev[['source', 'target']].copy()
            df_dev_good['labels'] = np.zeros(len(df_dev)).astype(int)
            df_dev_good = df_dev_good[:500]
            df_dev_bad = df_dev[['source', 'target_augmented']].copy()
            df_dev_bad = df_dev_bad.rename(columns={"target_augmented": "target"})
            df_dev_bad['labels'] = np.ones(len(df_dev)).astype(int)
            df_dev_bad = df_dev_bad[500:]
            df_dev = pd.concat((df_dev_good, df_dev_bad), ignore_index=True)
            df_dev = df_dev.sample(frac=1, random_state=1)

            df_test_good = df_test[['source', 'target']].copy()
            df_test_good['labels'] = np.zeros(len(df_test)).astype(int)
            df_test_good = df_test_good[:500]
            df_test_bad = df_test[['source', 'target_augmented']].copy()
            df_test_bad = df_test_bad.rename(columns={"target_augmented": "target"})
            df_test_bad['labels'] = np.ones(len(df_test)).astype(int)
            df_test_bad = df_test_bad[500:]
            df_test = pd.concat((df_test_good, df_test_bad), ignore_index=True)
            df_test = df_test.sample(frac=1, random_state=1)
            df_test['index'] = np.arange(0, len(df_test))

    if prep_for_training:
        test = df_test[['index', 'source', 'target']]
        train = df_train.rename(columns={'source': 'text_a', 'target': 'text_b'})
        dev = df_dev.rename(columns={'source': 'text_a', 'target': 'text_b'})
        test = test.rename(columns={'source': 'text_a', 'target': 'text_b'})
        return train, dev, test
    else:
        return df_train, df_dev, df_test


def get_sentence_length(df, source_or_target='target'):
    '''This method takes dataframes as input and assumes that the text columns
    are named text_a and text_b or original and translation. '''

    if source_or_target == 'target':
        if 'text_b' in df.columns:
            df['sentence_length'] = df['text_b'].str.split().str.len()
        else:
            df['sentence_length'] = df['translation'].str.split().str.len()

    else:
        if 'text_a' in df.columns:
            df['sentence_length'] = df['text_a'].str.split().str.len()
        else:
            df['sentence_length'] = df['original'].str.split().str.len()

    return df


def swap_sentence_pairs(df, shuffle_column = 'text_a'):

    df[shuffle_column] = df[shuffle_column].sample(frac=1, random_state=1).values

    return df