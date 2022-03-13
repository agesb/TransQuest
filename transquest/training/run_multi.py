import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
import torch
from torch.utils.data import Dataset
import requests
import tqdm
import regex
import scipy
import sklearn
import tokenizers
import sentencepiece
from nlp import Dataset
from tqdm import tqdm
import sys
import os
import errno
import shutil
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from pathlib import Path

# Set random seed and device
SEED = 1

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

GPU = True
if GPU:
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f'Using {device}')
print(torch.cuda.get_device_name(0))

# Pandas Display settings
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth', None)

from CODE.transquest.training.util.prep_data import load_MLQE_data, load_WikiMatrix_data, swap_sentence_pairs, get_sentence_length
from CODE.transquest.training.util.draw import draw_scatterplot_multitransquest, print_stat
from CODE.transquest.training.util.normalizer import fit, un_fit
from CODE.transquest.algo.sentence_level.multitransquest.evaluation import pearson_corr, spearman_corr, rmse
from CODE.transquest.algo.sentence_level.multitransquest.run_model import MultiTransQuestModel
from CODE.transquest.algo.sentence_level.multitransquest.grad_reversal import WeightGradientsFunc, WeightGradients, test_weight_gradients
from CODE.transquest.training.multitransquest_config import TEMP_DIRECTORY, DRIVE_FILE_ID, MODEL_NAME, \
    GOOGLE_DRIVE, multitransquest_config, MODEL_TYPE, SEED, RESULT_FILE, RESULT_IMAGE, SUBMISSION_FILE
from CODE.transquest.algo.sentence_level.multitransquest.utils import sweep_config_to_sweep_values


def train_MultiTransQuest(data, language,  n_heads, wandb_group=None, **kwargs):
    assert language in ['EN-DE', 'EN-ZH', 'ET-EN', 'RO-EN', 'SI-EN', 'NE-EN', 'RU-EN', 'MULTI']

    if "task_config" in kwargs:
        task_config = kwargs.get("task_config")
        multitransquest_config.update(task_config)

    print('I am working with', MODEL_TYPE, MODEL_NAME)

    if not os.path.exists(TEMP_DIRECTORY):
        os.makedirs(TEMP_DIRECTORY)

    # Unpack the datasets
    train_dataframes, dev_dataframes, test_dataframes, list_test_sentence_pairs = data

    # Normalise the labels
    for i in range(len(train_dataframes)):
        train_dataframes[i] = fit(train_dataframes[i], 'labels')
        dev_dataframes[i] = fit(dev_dataframes[i], 'labels')

    if multitransquest_config["evaluate_during_training"]:

        if multitransquest_config["n_fold"] > 1:

            from collections import defaultdict
            dev_predictions_dict = defaultdict(list)
            test_predictions_dict = defaultdict(list)
            ##dev_preds = np.zeros((len(dev), multitransquest_config["n_fold"]))
            # test_preds = np.zeros((len(test), multitransquest_config["n_fold"]))
            dev_preds = []
            test_preds = []
            for i in range(multitransquest_config["n_fold"]):

                if os.path.exists(multitransquest_config['output_dir']) and os.path.isdir(
                        multitransquest_config['output_dir']):
                    shutil.rmtree(multitransquest_config['output_dir'])

                model = MultiTransQuestModel(MODEL_TYPE, MODEL_NAME, wandb_group=wandb_group,
                                             use_cuda=torch.cuda.is_available(),
                                             args=multitransquest_config, **kwargs)

                train_dfs = []
                eval_dfs = []
                for df in train_dataframes:
                    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=SEED * i)
                    train_dfs.append(train_df)
                    eval_dfs.append(eval_df)

                model.train_model(train_dfs, eval_df=eval_dfs, multi_label=False)

                model = MultiTransQuestModel(MODEL_TYPE, multitransquest_config["best_model_dir"],
                                             wandb_group=wandb_group,
                                             use_cuda=torch.cuda.is_available(), args=multitransquest_config, **kwargs)

                for head in range(n_heads):
                    result, model_outputs, wrong_predictions = model.eval_model(dev_dataframes[head], curr_task=head,
                                                                                multi_label=False)

                    predictions, raw_outputs = model.predict(list_test_sentence_pairs[head], curr_task=head)

                    if multitransquest_config['num_labels'][head] == 1:
                        dev_predictions_dict[head].append(model_outputs)
                    else:
                        dev_predictions_dict[head].append(model_outputs[:, 0])

                    test_predictions_dict[head].append([predictions])

            list_test_results = []
            list_dev_results = []
            for head in range(n_heads):
                test_dataframes[head]['predictions'] = np.array(test_predictions_dict[head]).squeeze().mean(axis=0)
                dev_dataframes[head]['predictions'] = np.array(dev_predictions_dict[head]).squeeze().mean(axis=0)

                df_dev_results = un_fit(dev_dataframes[head], 'labels')
                df_dev_results = un_fit(dev_dataframes[head], 'predictions')
                df_test_results = un_fit(test_dataframes[head], 'predictions')
                df_dev_results.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False,
                                      encoding='utf-8')
                draw_scatterplot_multitransquest(df_dev_results, 'labels', 'predictions',
                                                 os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), language, curr_task=head)
                print_stat(df_dev_results, 'labels', 'predictions')
                list_test_results.append(df_test_results)
                list_dev_results.append(df_dev_results)

        else:

            model = MultiTransQuestModel(MODEL_TYPE, MODEL_NAME, wandb_group=wandb_group,
                                         use_cuda=torch.cuda.is_available(),
                                         args=multitransquest_config, **kwargs)

            train_dfs = []
            eval_dfs = []

            for df in train_dataframes:
                train_df, eval_df = train_test_split(df, test_size=0.1, random_state=SEED)
                train_dfs.append(train_df)
                eval_dfs.append(eval_df)
            model.train_model(train_dfs, eval_df=eval_dfs, multi_label=False)

            model = MultiTransQuestModel(MODEL_TYPE, multitransquest_config["best_model_dir"], wandb_group=wandb_group,
                                         use_cuda=torch.cuda.is_available(), args=multitransquest_config, **kwargs)

            list_test_results = []
            list_dev_results = []
            for head in range(n_heads):
                result, model_outputs, wrong_predictions = model.eval_model(dev_dataframes[head], curr_task=head,
                                                                            multi_label=False)

                predictions, raw_outputs = model.predict(list_test_sentence_pairs[head], curr_task=head)

                if multitransquest_config['num_labels'][head] == 1:
                    dev_dataframes[head]['predictions'] = model_outputs
                else:
                    dev_dataframes[head]['predictions'] = model_outputs[:, 0]

                test_dataframes[head]['predictions'] = predictions

                dev_head = un_fit(dev_dataframes[head], 'labels')
                dev_head = un_fit(dev_dataframes[head], 'predictions')
                test_head = un_fit(test_dataframes[head], 'predictions')
                dev_head.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False,
                                encoding='utf-8')
                draw_scatterplot_multitransquest(dev_head, 'labels', 'predictions',
                                                 os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), language, curr_task=head)
                print_stat(dev_head, 'labels', 'predictions')
                list_test_results.append(test_head)
                list_dev_results.append(dev_head)



    else:
        model = MultiTransQuestModel(MODEL_TYPE, MODEL_NAME, wandb_group=wandb_group,
                                     use_cuda=torch.cuda.is_available(),
                                     args=multitransquest_config, **kwargs)

        model.train_model(train_dataframes, multi_label=False)

        # Get evaluation and prediction per task
        list_test_results = []
        list_dev_results = []
        for head in range(n_heads):

            result, model_outputs, wrong_predictions = model.eval_model(dev_dataframes[head], curr_task=head,
                                                                        multi_label=False)

            predictions, raw_outputs = model.predict(list_test_sentence_pairs[head], curr_task=head)

            if multitransquest_config['num_labels'][head] == 1:
                dev_dataframes[head]['predictions'] = model_outputs
            else:
                dev_dataframes[head]['predictions'] = model_outputs[:, 0]

            test_dataframes[head]['predictions'] = predictions

            dev_head = un_fit(dev_dataframes[head], 'labels')
            dev_head = un_fit(dev_dataframes[head], 'predictions')
            test_head = un_fit(test_dataframes[head], 'predictions')
            dev_head.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False,
                            encoding='utf-8')
            draw_scatterplot_multitransquest(dev_head, 'labels', 'predictions',
                                             os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), language, curr_task=head)
            print_stat(dev_head, 'labels', 'predictions')
            list_test_results.append(test_head)
            list_dev_results.append(dev_head)
    return list_test_results, list_dev_results


def multitask_mixed_labels(language = 'EN-DE', labels = ['DA', 'HTER'], wandb_group=None, is_sweeping=False, **kwargs):

    if is_sweeping:
        with wandb.init(group=wandb_group) as run:

            sweep_config = wandb.config
            print('CONFIG', wandb.config)
            language = sweep_config['language']
            labels = sweep_config['labels']
            train_dataframes = []
            dev_dataframes = []
            test_dataframes = []
            list_test_sentence_pairs = []

            for label in labels:
                # Load train, dev and test data with the corresponding label
                train, dev, test = load_MLQE_data(language=language, label=label, prep_for_training=True)

                # Create test sentence pairs in the expected format
                test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

                # Store dataframes and test sentence pairs
                train_dataframes.append(train)
                dev_dataframes.append(dev)
                test_dataframes.append(test)
                list_test_sentence_pairs.append(test_sentence_pairs)

            data = [train_dataframes, dev_dataframes, test_dataframes, list_test_sentence_pairs]

            test_preds_per_task, dev_preds_per_task = train_MultiTransQuest(data, language, wandb_group,
                                                                            n_heads=len(train_dataframes))
    else:
        train_dataframes = []
        dev_dataframes = []
        test_dataframes = []
        list_test_sentence_pairs = []

        for label in labels:
            # Load train, dev and test data with the corresponding label
            train, dev, test = load_MLQE_data(language=language, label=label, prep_for_training=True)

            # Create test sentence pairs in the expected format
            test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

            # Store dataframes and test sentence pairs
            train_dataframes.append(train)
            dev_dataframes.append(dev)
            test_dataframes.append(test)
            list_test_sentence_pairs.append(test_sentence_pairs)

        data = [train_dataframes, dev_dataframes, test_dataframes, list_test_sentence_pairs]

        test_preds_per_task, dev_preds_per_task = train_MultiTransQuest(data, language, n_heads=len(train_dataframes), **kwargs)

    return test_preds_per_task, dev_preds_per_task


def multitask_mixed_languages(languages=["EN-DE", "RO-EN"], label= "DA", wandb_group="mixed_languages_multitask", is_sweeping=False, **kwargs):
    if is_sweeping:
        #wandb_group = SWEEP_CONFIG['parameters']['wandb_group']['values'][0]
        with wandb.init(group=wandb_group) as run:

            sweep_config = wandb.config
            print('CONFIG', wandb.config)
            languages = sweep_config['languages']
            label = sweep_config['label']

            train_dataframes = []
            dev_dataframes = []
            test_dataframes = []
            list_test_sentence_pairs = []

            for language in languages:
                train, dev, test = load_MLQE_data(language=language, label=label,
                                                  prep_for_training=True)
                # Create test sentence pairs in the expected format
                test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

                # Store dataframes and test sentence pairs
                train_dataframes.append(train)
                dev_dataframes.append(dev)
                test_dataframes.append(test)
                list_test_sentence_pairs.append(test_sentence_pairs)

            data = [train_dataframes, dev_dataframes, test_dataframes, list_test_sentence_pairs]

            test_preds_per_task, dev_preds_per_task = train_MultiTransQuest(data, language, wandb_group,
                                                                            n_heads=len(train_dataframes))

    else:
        train_dataframes = []
        dev_dataframes = []
        test_dataframes = []
        list_test_sentence_pairs = []

        for language in languages:
            train, dev, test = load_MLQE_data(language=language, label=label,
                                              prep_for_training=True)
            # Create test sentence pairs in the expected format
            test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

            # Store dataframes and test sentence pairs
            train_dataframes.append(train)
            dev_dataframes.append(dev)
            test_dataframes.append(test)
            list_test_sentence_pairs.append(test_sentence_pairs)

        data = [train_dataframes, dev_dataframes, test_dataframes, list_test_sentence_pairs]

        test_preds_per_task, dev_preds_per_task = train_MultiTransQuest(data, language, n_heads=len(train_dataframes), **kwargs)

    return test_preds_per_task, dev_preds_per_task


def multitask_augmented_wiki_data(language="EN-DE", wandb_group="aug_data_multitask", label='DA', is_sweeping=False, **kwargs):
    if is_sweeping:
        print(wandb_group)
        with wandb.init(group=wandb_group) as run:

            sweep_config = wandb.config
            print('CONFIG', wandb.config)
            language = sweep_config['language']
            print(language)
            label = sweep_config['label']

            train_dataframes = []
            dev_dataframes = []
            test_dataframes = []
            list_test_sentence_pairs = []

            data = ['MLQE', 'WikiMatrix_Binary_Classification']

            for dataset in data:
                if dataset == 'MLQE':
                    train, dev, test = load_MLQE_data(language=language, label=label, prep_for_training=True)

                elif dataset == 'WikiMatrix_Binary_Classification':
                    if language == 'EN-DE':
                        file = 'ende_9000_aug_custom_pipeline.tsv'
                    if language == 'EN-ZH':
                        file = 'wiki_enzh_9000.tsv'
                    train, dev, test = load_WikiMatrix_data(language=language, label=label,\
                                                            file=file, prep_for_training=True)

                else:
                    print('Please specify which method should be used to load this dataset')

                # Create test sentence pairs in the expected format
                test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

                # Store dataframes and test sentence pairs
                train_dataframes.append(train)
                dev_dataframes.append(dev)
                test_dataframes.append(test)
                list_test_sentence_pairs.append(test_sentence_pairs)

            data = [train_dataframes, dev_dataframes, test_dataframes, list_test_sentence_pairs]

            test_preds_per_task, dev_preds_per_task = train_MultiTransQuest(data, language,
                                                                            n_heads=len(train_dataframes))

    else:
        train_dataframes = []
        dev_dataframes = []
        test_dataframes = []
        list_test_sentence_pairs = []

        data = ['MLQE', 'WikiMatrix_Binary_Classification']

        for dataset in data:
            if dataset == 'MLQE':
                train, dev, test = load_MLQE_data(language=language, label=label, prep_for_training=True)

            elif dataset == 'WikiMatrix_Binary_Classification':
                if language == 'EN-DE':
                    file = 'ende_9000_aug_custom_pipeline.tsv'
                if language == 'EN-ZH':
                    file = 'wiki_enzh_9000.tsv'
                train, dev, test = load_WikiMatrix_data(language=language, label=label, \
                                                        file=file, prep_for_training=True)

            else:
                print('Please specify which method should be used to load this dataset')

            # Create test sentence pairs in the expected format
            test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

            # Store dataframes and test sentence pairs
            train_dataframes.append(train)
            dev_dataframes.append(dev)
            test_dataframes.append(test)
            list_test_sentence_pairs.append(test_sentence_pairs)

        data = [train_dataframes, dev_dataframes, test_dataframes, list_test_sentence_pairs]

        test_preds_per_task, dev_preds_per_task = train_MultiTransQuest(data, language,
                                                                        n_heads=len(train_dataframes), **kwargs)
    return test_preds_per_task, dev_preds_per_task


def multitask_shuffled_MLQE_data(language="EN-DE", wandb_group="shu_MLQE_data_multitask", is_sweeping=False, **kwargs):
    if is_sweeping:
        with wandb.init(group=wandb_group) as run:

            sweep_config = wandb.config
            print('CONFIG', wandb.config)
            language = sweep_config['language']
            print(language)
            label = sweep_config['label']

            train_dataframes = []
            dev_dataframes = []
            test_dataframes = []
            list_test_sentence_pairs = []

            data = ['MLQE', 'MLQE_shuffle']

            for dataset in data:
                train, dev, test = load_MLQE_data(language=language, label='DA', prep_for_training=True)

                if dataset == 'MLQE_shuffle':
                    train["shuffled_text_a"] = train["text_a"].sample(frac=1, random_state=1).values
                    df_train_good = train[['text_a', 'text_b']].copy()
                    df_train_good['labels'] = np.ones(len(train)).astype(int)
                    df_train_good = df_train_good[:3500]
                    df_train_bad = train[['shuffled_text_a', 'text_b']].copy()
                    df_train_bad['labels'] = np.zeros(len(train)).astype(int)
                    df_train_bad = df_train_bad.rename(columns={"shuffled_text_a": "text_a"})
                    df_train_bad = df_train_bad[3500:]
                    train = pd.concat((df_train_good, df_train_bad), ignore_index=True)
                    train = train.sample(frac=1, random_state=1)

                    dev["shuffled_text_a"] = dev["text_a"].sample(frac=1, random_state=1).values
                    df_dev_good = dev[['text_a', 'text_b']].copy()
                    df_dev_good['labels'] = np.ones(len(dev)).astype(int)
                    df_dev_good = df_dev_good[:500]
                    df_dev_bad = dev[['shuffled_text_a', 'text_b']].copy()
                    df_dev_bad = df_dev_bad.rename(columns={"shuffled_text_a": "text_a"})
                    df_dev_bad['labels'] = np.zeros(len(dev)).astype(int)
                    df_dev_bad = df_dev_bad[500:]
                    dev = pd.concat((df_dev_good, df_dev_bad), ignore_index=True)
                    dev = dev.sample(frac=1, random_state=1)

                    test["shuffled_text_a"] = test["text_a"].sample(frac=1, random_state=1).values
                    df_test_good = test[['text_a', 'text_b']].copy()
                    df_test_good['labels'] = np.ones(len(test)).astype(int)
                    df_test_good = df_test_good[:500]
                    df_test_bad = test[['shuffled_text_a', 'text_b']].copy()
                    df_test_bad = df_test_bad.rename(columns={"shuffled_text_a": "text_a"})
                    df_test_bad['labels'] = np.zeros(len(test)).astype(int)
                    df_test_bad = df_test_bad[500:]
                    test = pd.concat((df_test_good, df_test_bad), ignore_index=True)
                    test = test.sample(frac=1, random_state=1)
                    test['index'] = np.arange(0, len(test))



                # Create test sentence pairs in the expected format
                test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

                # Store dataframes and test sentence pairs
                train_dataframes.append(train)
                dev_dataframes.append(dev)
                test_dataframes.append(test)
                list_test_sentence_pairs.append(test_sentence_pairs)

            data = [train_dataframes, dev_dataframes, test_dataframes, list_test_sentence_pairs]

            test_preds_per_task, dev_preds_per_task = train_MultiTransQuest(data, language, wandb_group,
                                                                            n_heads=len(train_dataframes))

    else:
        train_dataframes = []
        dev_dataframes = []
        test_dataframes = []
        list_test_sentence_pairs = []

        data = ['MLQE', 'MLQE_shuffle']

        for dataset in data:
            train, dev, test = load_MLQE_data(language=language, label='DA', prep_for_training=True)

            if dataset == 'MLQE_shuffle':
                train["shuffled_text_a"] = train["text_a"].sample(frac=1, random_state=1).values
                df_train_good = train[['text_a', 'text_b']].copy()
                df_train_good['labels'] = np.ones(len(train)).astype(int)
                df_train_good = df_train_good[:3500]
                df_train_bad = train[['shuffled_text_a', 'text_b']].copy()
                df_train_bad['labels'] = np.zeros(len(train)).astype(int)
                df_train_bad = df_train_bad.rename(columns={"shuffled_text_a": "text_a"})
                df_train_bad = df_train_bad[3500:]
                train = pd.concat((df_train_good, df_train_bad), ignore_index=True)
                train = train.sample(frac=1, random_state=1)

                dev["shuffled_text_a"] = dev["text_a"].sample(frac=1, random_state=1).values
                df_dev_good = dev[['text_a', 'text_b']].copy()
                df_dev_good['labels'] = np.ones(len(dev)).astype(int)
                df_dev_good = df_dev_good[:500]
                df_dev_bad = dev[['shuffled_text_a', 'text_b']].copy()
                df_dev_bad = df_dev_bad.rename(columns={"shuffled_text_a": "text_a"})
                df_dev_bad['labels'] = np.zeros(len(dev)).astype(int)
                df_dev_bad = df_dev_bad[500:]
                dev = pd.concat((df_dev_good, df_dev_bad), ignore_index=True)
                dev = dev.sample(frac=1, random_state=1)

                test["shuffled_text_a"] = test["text_a"].sample(frac=1, random_state=1).values
                df_test_good = test[['text_a', 'text_b']].copy()
                df_test_good['labels'] = np.ones(len(test)).astype(int)
                df_test_good = df_test_good[:500]
                df_test_bad = test[['shuffled_text_a', 'text_b']].copy()
                df_test_bad = df_test_bad.rename(columns={"shuffled_text_a": "text_a"})
                df_test_bad['labels'] = np.zeros(len(test)).astype(int)
                df_test_bad = df_test_bad[500:]
                test = pd.concat((df_test_good, df_test_bad), ignore_index=True)
                test = test.sample(frac=1, random_state=1)
                test['index'] = np.arange(0, len(test))

            # Create test sentence pairs in the expected format
            test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

            # Store dataframes and test sentence pairs
            train_dataframes.append(train)
            dev_dataframes.append(dev)
            test_dataframes.append(test)
            list_test_sentence_pairs.append(test_sentence_pairs)

        data = [train_dataframes, dev_dataframes, test_dataframes, list_test_sentence_pairs]

        test_preds_per_task, dev_preds_per_task = train_MultiTransQuest(data, language, wandb_group,
                                                                        n_heads=len(train_dataframes), **kwargs)

    return test_preds_per_task, dev_preds_per_task



def multitask_partial_input(language="EN-DE", label="DA", wandb_group="adv_partial_input", is_sweeping=False, **kwargs):
    if is_sweeping:
        with wandb.init(group=wandb_group) as run:

            sweep_config = wandb.config
            print('CONFIG', wandb.config)

            train_dataframes = []
            dev_dataframes = []
            test_dataframes = []
            list_test_sentence_pairs = []

            language = sweep_config['language']
            label = sweep_config['label']

            assert label in ['DA', 'HTER']

            partial_inputs = ['both', 'target']

            for partial_input in partial_inputs:

                # Load train, dev and test data with the corresponding label
                train, dev, test = load_MLQE_data(language=language, label=label, prep_for_training=True)

                if partial_input == 'source':
                    train = train.drop('text_b', axis=1)
                    train = train.rename(columns={"text_a": "text"})
                    dev = dev.drop('text_b', axis=1)
                    dev = dev.rename(columns={"text_a": "text"})
                    test = test.drop('text_b', axis=1)
                    test = test.rename(columns={"text_a": "text"})
                    test_sentence_pairs = test['text'].to_list()
                elif partial_input == 'target':
                    train = train.drop('text_a', axis=1)
                    train = train.rename(columns={"text_b": "text"})
                    dev = dev.drop('text_b', axis=1)
                    dev = dev.rename(columns={"text_a": "text"})
                    test = test.drop('text_b', axis=1)
                    test = test.rename(columns={"text_a": "text"})
                    test_sentence_pairs = test['text'].to_list()

                else:
                    test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

                # Store dataframes and test sentence pairs
                train_dataframes.append(train)
                dev_dataframes.append(dev)
                test_dataframes.append(test)
                list_test_sentence_pairs.append(test_sentence_pairs)

            data = [train_dataframes, dev_dataframes, test_dataframes, list_test_sentence_pairs]

            test_preds_per_task, dev_preds_per_task = train_MultiTransQuest(data, language, wandb_group,
                                                                            n_heads=len(train_dataframes),
                                                                            sweep_config=sweep_config)

    else:

        train_dataframes = []
        dev_dataframes = []
        test_dataframes = []
        list_test_sentence_pairs = []

        assert label in ['DA', 'HTER']

        partial_inputs = ['both', 'target']

        for partial_input in partial_inputs:

            # Load train, dev and test data with the corresponding label
            train, dev, test = load_MLQE_data(language=language, label=label, prep_for_training=True)

            if partial_input == 'source':
                train = train.drop('text_b', axis=1)
                train = train.rename(columns={"text_a": "text"})
                dev = dev.drop('text_b', axis=1)
                dev = dev.rename(columns={"text_a": "text"})
                test = test.drop('text_b', axis=1)
                test = test.rename(columns={"text_a": "text"})
                test_sentence_pairs = test['text'].to_list()
            elif partial_input == 'target':
                train = train.drop('text_a', axis=1)
                train = train.rename(columns={"text_b": "text"})
                dev = dev.drop('text_b', axis=1)
                dev = dev.rename(columns={"text_a": "text"})
                test = test.drop('text_b', axis=1)
                test = test.rename(columns={"text_a": "text"})
                test_sentence_pairs = test['text'].to_list()

            else:
                test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

            # Store dataframes and test sentence pairs
            train_dataframes.append(train)
            dev_dataframes.append(dev)
            test_dataframes.append(test)
            list_test_sentence_pairs.append(test_sentence_pairs)

        data = [train_dataframes, dev_dataframes, test_dataframes, list_test_sentence_pairs]

        test_preds_per_task, dev_preds_per_task = train_MultiTransQuest(data, language,
                                                                        n_heads=len(train_dataframes), **kwargs)
    return test_preds_per_task, dev_preds_per_task


def multitask_sentence_length(language='EN-DE', label="DA", sentence_length_input='text_a', wandb_group="adv-sentence-length",
                              is_sweeping=False, **kwargs):
    if is_sweeping:
        with wandb.init(group=wandb_group) as run:
            sweep_config = wandb.config
            print('CONFIG', wandb.config)
            language = sweep_config['language']
            label = sweep_config['label']
            sentence_length_input = sweep_config['sentence_length_input']

            config_defaults = multitransquest_config
            run.config.setdefaults(config_defaults)

            print(wandb_group)
            print('CONFIG', wandb.config)

            train_dataframes = []
            dev_dataframes = []
            test_dataframes = []
            list_test_sentence_pairs = []

            assert label in ['DA', 'HTER']
            assert sentence_length_input in ['text_a', 'text_b']

            for use_sentence_length in [False, True]:
                # Load train, dev and test data with the corresponding label
                train, dev, test = load_MLQE_data(language=language, label=label, prep_for_training=True)

                if use_sentence_length:
                    train['labels'] = train[sentence_length_input].str.split().str.len()
                    dev['labels'] = dev[sentence_length_input].str.split().str.len()
                    test['labels'] = test[sentence_length_input].str.split().str.len()
                test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

                # Store dataframes and test sentence pairs
                train_dataframes.append(train)
                dev_dataframes.append(dev)
                test_dataframes.append(test)
                list_test_sentence_pairs.append(test_sentence_pairs)

            data = [train_dataframes, dev_dataframes, test_dataframes, list_test_sentence_pairs]

            test_preds_per_task, dev_preds_per_task = train_MultiTransQuest(data, language, wandb_group,
                                                                            n_heads=len(train_dataframes),
                                                                            sweep_config=sweep_config)
    else:
        train_dataframes = []
        dev_dataframes = []
        test_dataframes = []
        list_test_sentence_pairs = []

        assert label in ['DA', 'HTER']
        assert sentence_length_input in ['text_a', 'text_b']

        for use_sentence_length in [False, True]:
            # Load train, dev and test data with the corresponding label
            train, dev, test = load_MLQE_data(language=language, label=label, prep_for_training=True)

            if use_sentence_length:
                train['labels'] = train[sentence_length_input].str.split().str.len()
                dev['labels'] = dev[sentence_length_input].str.split().str.len()
                test['labels'] = test[sentence_length_input].str.split().str.len()
            test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

            # Store dataframes and test sentence pairs
            train_dataframes.append(train)
            dev_dataframes.append(dev)
            test_dataframes.append(test)
            list_test_sentence_pairs.append(test_sentence_pairs)

        data = [train_dataframes, dev_dataframes, test_dataframes, list_test_sentence_pairs]

        test_preds_per_task, dev_preds_per_task = train_MultiTransQuest(data, language,
                                                                        n_heads=len(train_dataframes), **kwargs)

    return test_preds_per_task, dev_preds_per_task


# Inference: Using the trained model for predictions

def predict_MultiTransQuest(model_language='EN-DE', language="EN-DE", evaluation_type="DA", save_results=True, dataset="test", data='MLQE', task_number=0, task="regression", aux_type="multilanguage", experiment='', partial_input=None, shuffle_column=None, **kwargs):

    if "task_config" in kwargs:
        task_config = kwargs.get("task_config")
        multitransquest_config.update(task_config)

    if data == 'wikimatrix':
        if model_language == 'EN-DE':
            file = "ende_9000_aug_custom_pipeline.tsv"
        #elif model_language == 'EN-ZH':
        #    file = "enzh_9000_aug_custom_pipeline.tsv"
        else:
            print('Please specify WikiMatrix data file')
        train, dev, test = load_WikiMatrix_data(file=file, language=language, prep_for_training=True)

    else:
        if evaluation_type == 'DA':
            train, dev, test = load_MLQE_data(language=language, label='DA', prep_for_training=True)
        else:
            train, dev, test = load_MLQE_data(language=language, label='HTER', prep_for_training=True)

    if dataset == 'test':
        reference=test

        if partial_input == "target":
            test = test.drop('text_a', axis=1)
            test = test.rename(columns={"text_b": "text"})
            test_sentence_pairs = test['text'].to_list()

        elif partial_input == "source":
            test = test.drop('text_b', axis=1)
            test = test.rename(columns={"text_a": "text"})
            test_sentence_pairs = test['text'].to_list()

        else:
            if shuffle_column is not None:
                assert shuffle_column in ['text_a', 'text_b']
                test = swap_sentence_pairs(test, shuffle_column=shuffle_column)

            test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

    elif dataset == 'dev':
        reference = dev
        if partial_input == "target":
            dev = dev.drop('text_a', axis=1)
            dev = dev.rename(columns={"text_b": "text"})
            test_sentence_pairs = dev['text'].to_list()

        elif partial_input == "source":
            dev = dev.drop('text_b', axis=1)
            dev = dev.rename(columns={"text_a": "text"})
            test_sentence_pairs = dev['text'].to_list()

        else:
            if shuffle_column is not None:
                assert shuffle_column in ['text_a', 'text_b']
                dev = swap_sentence_pairs(dev, shuffle_column=shuffle_column)

            test_sentence_pairs = list(map(list, zip(dev['text_a'].to_list(), dev['text_b'].to_list())))
    else:
        print('Please specify the dataset used for predictions (either dev or test)')


    model = MultiTransQuestModel(MODEL_TYPE, multitransquest_config["best_model_dir"],
                                 use_cuda=torch.cuda.is_available(), wandb_group="aug_data_multitask",
                                 args=multitransquest_config, **kwargs)

    predictions, raw_outputs = model.predict(test_sentence_pairs, curr_task=task_number)

    if data == 'wikimatrix':
        df_pred = pd.DataFrame(predictions, columns=['prediction'])

    else:
        df_pred = pd.DataFrame(predictions, columns=['Predicted_' + evaluation_type])

        if save_results:
            if experiment == "FINAL":
                seed = task_config['manual_seed']
                print("Seed:",seed)
                if language == model_language:
                    if partial_input == "source":
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/' + aux_type + '/' + experiment + '/'  + dataset + '_data/' + str(seed) + '_pred_' + dataset + '_' + evaluation_type + '_multi_partial_source.csv')
                    elif partial_input == "target":
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/' + aux_type + '/' + experiment + '/'  + dataset + '_data/' + str(seed) + '_pred_' + dataset + '_' + evaluation_type + '_multi_partial_target.csv')
                    elif shuffle_column == "text_a":
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/' + aux_type + '/' + experiment + '/'  + dataset + '_data/' + str(seed) + '_pred_' + dataset + '_' + evaluation_type + '_multi_shuffle_source.csv')
                    elif shuffle_column == "text_b":
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/' + aux_type + '/' + experiment + '/'  + dataset + '_data/' + str(seed) + '_pred_' + dataset + '_' + evaluation_type + '_multi_shuffle_target.csv')

                    else:
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/' + aux_type + '/' + experiment + '/'  + dataset + '_data/' + str(seed) + '_pred_' + dataset + '_' + evaluation_type + '_multi.csv')

                # Out of domain predictions
                else:
                    if partial_input == "source":
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/' + aux_type + '/' + experiment + '/' + str(seed) + '_' + dataset + '_data/OOD_' + language + '_pred_' + dataset + '_' + evaluation_type + '_multi_partial_source.csv')
                    elif partial_input == "target":
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/' + aux_type + '/' + experiment + '/' + str(seed) + '_' + dataset + '_data/OOD_' + language + '_pred_' + dataset + '_' + evaluation_type + '_multi_partial_target.csv')
                    elif shuffle_column == "text_a":
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/' + aux_type + '/' + experiment + '/' + str(seed) + '_' + dataset + '_data/OOD_' + language + '_pred_' + dataset + '_' + evaluation_type + '_multi_shuffle_source.csv')
                    elif shuffle_column == "text_b":
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/' + aux_type + '/' + experiment + '/' + str(seed) + '_' + dataset + '_data/OOD_' + language + '_pred_' + dataset + '_' + evaluation_type + '_multi_shuffle_target.csv')

                    else:
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/' + aux_type + '/' + experiment + '/'+ str(seed) + '_' + dataset + '_data/OOD_' + language + '_pred_' + dataset + '_' + evaluation_type + '_multi.csv')

            else:
                if language == model_language:
                    if partial_input == "source":
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/'+aux_type+'/' +experiment+'/' + dataset + '_data/pred_' + dataset + '_' + evaluation_type + '_multi_partial_source.csv')
                    elif partial_input == "target":
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/'+aux_type+'/' +experiment+'/' + dataset + '_data/pred_' + dataset + '_' + evaluation_type + '_multi_partial_target.csv')
                    elif shuffle_column == "text_a":
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/'+aux_type+'/' +experiment+'/' + dataset + '_data/pred_' + dataset + '_' + evaluation_type + '_multi_shuffle_source.csv')
                    elif shuffle_column == "text_b":
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/'+aux_type+'/' +experiment+'/' + dataset + '_data/pred_' + dataset + '_' + evaluation_type + '_multi_shuffle_target.csv')

                    else:
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/'+aux_type+'/' +experiment+'/' + dataset + '_data/pred_' + dataset + '_' + evaluation_type + '_multi.csv')

                # Out of domain predictions
                else:
                    if partial_input == "source":
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/'+aux_type+'/' +experiment+'/' + dataset + '_data/OOD_' + language + '_pred_'  + dataset + '_' + evaluation_type + '_multi_partial_source.csv')
                    elif partial_input == "target":
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/'+aux_type+'/' +experiment+'/' + dataset + '_data/OOD_' + language + '_pred_'  + dataset + '_' + evaluation_type + '_multi_partial_target.csv')
                    elif shuffle_column == "text_a":
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/'+aux_type+'/' +experiment+'/' + dataset + '_data/OOD_' + language + '_pred_'  + dataset + '_' + evaluation_type + '_multi_shuffle_source.csv')
                    elif shuffle_column == "text_b":
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/'+aux_type+'/' +experiment+'/' + dataset + '_data/OOD_' + language + '_pred_' + dataset + '_' + evaluation_type + '_multi_shuffle_target.csv')

                    else:
                        df_pred.to_csv(
                            'DATA/MLQE-PE/' + model_language + '/predictions/multitask/'+aux_type+'/' +experiment+'/' + dataset + '_data/OOD_' + language + '_pred_' + dataset + '_' + evaluation_type + '_multi.csv')

    return df_pred, reference


def main():
    sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project="multi-transquest")
    wandb.agent(sweep_id, function=multitask_augmented_wiki_data, count=20)

    model_language = "EN-DE"
    label = "DA"
    languages = ['EN-DE', 'EN-ZH', 'RO-EN', 'ET-EN', 'SI-EN', 'NE-EN', 'RU-EN']
    val_sets = ["test", "dev"]

    for val_set in val_sets:
        for val_language in languages:
            # Out of domain predictions
            if model_language != val_language:
                predict_MultiTransQuest(language=val_language, model_language=model_language, dataset=val_set, aux_type="aug",
                                       evaluation_type=label, data='MLQE', task='regression', partial_input=False)

            if val_language == model_language:
                # Predict on both sentences
                predict_MultiTransQuest(language=val_language, model_language=model_language, dataset=val_set, aux_type="aug",
                                       evaluation_type=label, data='MLQE', task='regression')

                # Predict on partial input and shuffled datasets
                predict_MultiTransQuest(language=val_language, model_language=model_language, dataset=val_set, aux_type="aug",
                                       evaluation_type=label, data='MLQE', task='regression', partial_input="source", )
                predict_MultiTransQuest(language=val_language, model_language=model_language, dataset=val_set, aux_type="aug",
                                       evaluation_type=label, data='MLQE', task='regression', partial_input="target")
                predict_MultiTransQuest(language=val_language, model_language=model_language, dataset=val_set, aux_type="aug",
                                       evaluation_type=label, data='MLQE', task='regression', shuffle_column="text_a")
                predict_MultiTransQuest(language=val_language, model_language=model_language, dataset=val_set, aux_type="aug",
                                       evaluation_type=label, data='MLQE', task='regression', shuffle_column="text_b")

if __name__ == "__main__":
    main()

