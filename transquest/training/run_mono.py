

# Import required libraries

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
from CODE.transquest.training.util.draw import draw_scatterplot, print_stat
from CODE.transquest.training.util.normalizer import fit, un_fit
from CODE.transquest.algo.sentence_level.monotransquest.evaluation import pearson_corr, spearman_corr, rmse
from CODE.transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel
from CODE.transquest.algo.sentence_level.monotransquest.models.lin_sentence_length_model import SentenceLengthBiasNet

from CODE.transquest.training.monotransquest_config import TEMP_DIRECTORY, DRIVE_FILE_ID, MODEL_NAME, \
    GOOGLE_DRIVE, monotransquest_config, MODEL_TYPE, SEED, RESULT_FILE, RESULT_IMAGE, SUBMISSION_FILE

def train_TransQuest(language='ET-EN', data='MLQE', task='regression', label='HTER', partial=False, focal_loss=False, **kwargs):
    assert language in ['EN-DE', 'EN-ZH', 'ET-EN', 'RO-EN', 'SI-EN', 'NE-EN', 'RU-EN']
    assert data in ['MLQE', 'WIKIMATRIX']
    assert task in ['regression', 'binary-classification']

    if "task_config" in kwargs:
        task_config = kwargs.get("task_config")
        monotransquest_config.update(task_config)

    print('I am working with', MODEL_TYPE, MODEL_NAME)

    if not os.path.exists(TEMP_DIRECTORY):
        os.makedirs(TEMP_DIRECTORY)

    if data == 'MLQE':
        train, dev, test = load_MLQE_data(language=language, label=label, prep_for_training=True)

        if partial:
            train = train.drop('text_a', axis=1)
            train = train.rename(columns={"text_b": "text"})

        if task == 'regression-sentence-length':
            train = get_sentence_length(train, df_type="train", prep_for_training=True)
            dev = get_sentence_length(dev, df_type="dev", prep_for_training=True)
            test = get_sentence_length(test, df_type="test", prep_for_training=True)

    else:
        if language == 'EN-DE':
            file = "ende_9000_aug_custom_pipeline.tsv"
        else:
            print('Please specify WikiMatrix data file')
        train, dev, test = load_WikiMatrix_data(file=file, language=language, prep_for_training=True)

    index = test['index'].to_list()
    test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

    train = fit(train, 'labels')
    dev = fit(dev, 'labels')

    assert (len(index) == 1000)
    if monotransquest_config["evaluate_during_training"]:
        if monotransquest_config["n_fold"] > 1:
            dev_preds = np.zeros((len(dev), monotransquest_config["n_fold"]))
            test_preds = np.zeros((len(test), monotransquest_config["n_fold"]))
            for i in range(monotransquest_config["n_fold"]):

                if os.path.exists(monotransquest_config['output_dir']) and os.path.isdir(
                        monotransquest_config['output_dir']):
                    shutil.rmtree(monotransquest_config['output_dir'])

                if task == 'regression':
                    model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1,
                                                use_cuda=torch.cuda.is_available(),
                                                args=monotransquest_config, **kwargs)
                    train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
                    model.train_model(train_df, focal_loss=focal_loss, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                                      mae=mean_absolute_error)

                    model = MonoTransQuestModel(MODEL_TYPE, monotransquest_config["best_model_dir"],
                                                num_labels=1,
                                                use_cuda=torch.cuda.is_available(), args=monotransquest_config, **kwargs)
                    result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
                                                                                spearman_corr=spearman_corr,
                                                                                mae=mean_absolute_error)
                    predictions, raw_outputs = model.predict(test_sentence_pairs)
                    dev_preds[:, i] = model_outputs
                    test_preds[:, i] = predictions

                elif task == 'binary-classification':
                    model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=2,
                                                use_cuda=torch.cuda.is_available(),
                                                args=monotransquest_config, **kwargs)
                    train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
                    model.train_model(train_df, eval_df=eval_df, multi_label=False)
                    model = MonoTransQuestModel(MODEL_TYPE, monotransquest_config["best_model_dir"],
                                                num_labels=2,
                                                use_cuda=torch.cuda.is_available(), args=monotransquest_config, **kwargs)
                    result, model_outputs, wrong_predictions = model.eval_model(dev, multi_label=False)

                    predictions, raw_outputs = model.predict(test_sentence_pairs)
                    dev_preds[:, i] = model_outputs[:, 0]
                    test_preds[:, i] = predictions

                else:
                    print('Task is not defined')

            dev['predictions'] = dev_preds.mean(axis=1)
            test['predictions'] = test_preds.mean(axis=1)

        else:
            if task == 'regression':
                model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1,
                                            use_cuda=torch.cuda.is_available(),
                                            args=monotransquest_config, **kwargs)
                train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
                model.train_model(train_df, eval_df=eval_df, focal_loss=focal_loss, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                                  mae=mean_absolute_error)
                model = MonoTransQuestModel(MODEL_TYPE, monotransquest_config["best_model_dir"],
                                            num_labels=1,
                                            use_cuda=torch.cuda.is_available(), args=monotransquest_config, **kwargs)
                result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
                                                                            spearman_corr=spearman_corr,
                                                                            mae=mean_absolute_error)
                predictions, raw_outputs = model.predict(test_sentence_pairs)
                dev['predictions'] = model_outputs
                test['predictions'] = predictions

            elif task == 'binary-classification':
                model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=2,
                                            use_cuda=torch.cuda.is_available(),
                                            args=monotransquest_config, **kwargs)
                train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
                model.train_model(train_df, eval_df=eval_df, multi_label=False)
                model = MonoTransQuestModel(MODEL_TYPE, monotransquest_config["best_model_dir"],
                                            num_labels=2,
                                            use_cuda=torch.cuda.is_available(), args=monotransquest_config, **kwargs)
                result, model_outputs, wrong_predictions = model.eval_model(dev, multi_label=False)

                predictions, raw_outputs = model.predict(test_sentence_pairs)
                dev['predictions'] = model_outputs[:, 0]
                test['predictions'] = predictions

            else:
                print('Task is not defined')



    else:
        if task == 'regression':
            model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME,  num_labels=1,
                                        use_cuda=torch.cuda.is_available(),
                                        args=monotransquest_config, **kwargs)
            model.train_model(train, focal_loss=focal_loss, pearson_corr=pearson_corr, spearman_corr=spearman_corr, mae=mean_absolute_error)
            result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
                                                                        spearman_corr=spearman_corr,
                                                                        mae=mean_absolute_error)
            predictions, raw_outputs = model.predict(test_sentence_pairs)
            dev['predictions'] = model_outputs
            test['predictions'] = predictions

        elif task == 'binary-classification':
            model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=2,
                                        use_cuda=torch.cuda.is_available(),
                                        args=monotransquest_config, **kwargs)
            model.train_model(train, multi_label=False)
            result, model_outputs, wrong_predictions = model.eval_model(dev, multi_label=False)
            predictions, raw_outputs = model.predict(test_sentence_pairs)
            dev['predictions'] = model_outputs[:, 0]
            test['predictions'] = predictions


        else:
            print('Task is not defined')

    dev = un_fit(dev, 'labels')
    dev = un_fit(dev, 'predictions')
    test = un_fit(test, 'predictions')
    dev.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
    draw_scatterplot(dev, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), language)
    print_stat(dev, 'labels', 'predictions')
    # format_submission(df=test, index=index, language_pair="en-de", method="TransQuest",
    #                 path=os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE))

    return dev, test

def train_benchmark(language = "EN-DE", data='MLQE', task='regression', label='HTER', is_sweeping = False, **kwargs):

    if "task_config" in kwargs:
        task_config = kwargs.get("task_config")
        monotransquest_config.update(task_config)

    if is_sweeping:
        wandb_group = SWEEP_CONFIG['parameters']['wandb_group']['values'][0]
        with wandb.init(group = "wikimatrix-classification") as run:

            sweep_config = wandb.config
            print('CONFIG (no defaults)',wandb.config)

            language = sweep_config['language']
            label = sweep_config['label']
            task = sweep_config['task']
            data = sweep_config['data']

            config_defaults = monotransquest_config
            run.config.setdefaults(config_defaults)

            print(wandb_group)
            print('CONFIG',wandb.config)

            dev, test = train_TransQuest(language=language, data=data, task=task, label=label, sweep_config = sweep_config)

    else:
        dev, test = train_TransQuest(language, label, **kwargs)

    return dev, test


def train_MonoTransQuestFocalLoss(language='EN-DE', bias_model_type="partial_input_target", label='DA', **kwargs):
    assert language in ['EN-DE', 'EN-ZH', 'ET-EN', 'RO-EN', 'SI-EN', 'NE-EN', 'RU-EN']

    if "task_config" in kwargs:
        task_config = kwargs.get("task_config")
        monotransquest_config.update(task_config)

    print('I am working with', MODEL_TYPE, MODEL_NAME)

    if not os.path.exists(TEMP_DIRECTORY):
        os.makedirs(TEMP_DIRECTORY)

    train, dev, test = load_MLQE_data(language=language, label=label, prep_for_training=True)
    index = test['index'].to_list()
    test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))
    train = fit(train, 'labels')
    dev = fit(dev, 'labels')

    # Get the data for the bias model
    if bias_model_type == "partial_input_target":
        dev_preds_bias_model = np.zeros((len(dev), monotransquest_config["n_fold"]))
        test_preds_bias_model = np.zeros((len(test), monotransquest_config["n_fold"]))
        train_bias_model = train.drop('text_a', axis=1)
        train_bias_model = train_bias_model.rename(columns={"text_b": "text"})
        dev_bias_model = dev.drop('text_a', axis=1)
        dev_bias_model = dev_bias_model.rename(columns={"text_b": "text"})
        test_bias_model = test.drop('text_a', axis=1)
        test_bias_model = test_bias_model.rename(columns={"text_b": "text"})
        test_sentence_pairs_bias_model = test_bias_model['text'].to_list()
        train_bias_model = fit(train_bias_model, 'labels')
        dev_bias_model = fit(dev_bias_model, 'labels')

    elif bias_model_type == "partial_input_source":
        dev_preds_bias_model = np.zeros((len(dev), monotransquest_config["n_fold"]))
        test_preds_bias_model = np.zeros((len(test), monotransquest_config["n_fold"]))
        train_bias_model = train.drop('text_b', axis=1)
        train_bias_model = train_bias_model.rename(columns={"text_a": "text"})
        dev_bias_model = dev.drop('text_b', axis=1)
        dev_bias_model = dev_bias_model.rename(columns={"text_a": "text"})
        test_bias_model = test.drop('text_b', axis=1)
        test_bias_model = test_bias_model.rename(columns={"text_a": "text"})
        test_sentence_pairs_bias_model = test_bias_model['text'].to_list()
        train_bias_model = fit(train_bias_model, 'labels')
        dev_bias_model = fit(dev_bias_model, 'labels')
        print(train_bias_model['text'][0])

    elif bias_model_type == "sentence_length":
        train_bias_model = get_sentence_length(train)
        dev_bias_model = get_sentence_length(dev)
        test_bias_model = get_sentence_length(test)

        test_sentence_pairs_bias_model = test_bias_model[['index', 'sentence_length']]

    # Store the dataframes for the main model and the bias model
    train_dfs = [train, train_bias_model]
    print(train.columns)
    print(train_bias_model.columns)
    dev_dfs = [dev, dev_bias_model]
    test_dfs = [test, test_bias_model]
    all_test_sentence_pairs = [test_sentence_pairs, test_sentence_pairs_bias_model]

    assert (len(index) == 1000)
    if monotransquest_config["evaluate_during_training"]:
        if monotransquest_config["n_fold"] > 1:
            dev_preds = np.zeros((len(dev), monotransquest_config["n_fold"]))
            test_preds = np.zeros((len(test), monotransquest_config["n_fold"]))
            # dev_preds_bias_model = np.zeros((len(dev), monotransquest_config["n_fold"]))
            # test_preds_bias_model = np.zeros((len(test), monotransquest_config["n_fold"]))

            for i in range(monotransquest_config["n_fold"]):

                if os.path.exists(monotransquest_config['output_dir']) and os.path.isdir(
                        monotransquest_config['output_dir']):
                    shutil.rmtree(monotransquest_config['output_dir'])

                model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                                            args=monotransquest_config, **kwargs)

                # Initialize the bias model
                if bias_model_type == "partial_input_target" or bias_model_type == "partial_input_source":
                    bias_model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1,
                                                     use_cuda=torch.cuda.is_available(),
                                                     args=monotransquest_config, **kwargs)
                    bias_type = 'partial_input'

                elif bias_model_type == 'sentence_length':
                    bias_model = SentenceLengthBiasNet()
                    bias_type = 'sentence_length'
                else:
                    print('Please specify the bias model type.')

                # Get the train and eval dataset
                split_train_dfs = []
                split_eval_dfs = []
                for train_df in train_dfs:
                    train_df, eval_df = train_test_split(train_df, test_size=0.1, random_state=SEED * i)
                    split_train_dfs.append(train_df)
                    split_eval_dfs.append(eval_df)

                model.train_model_with_focal_loss(bias_model=bias_model, bias_type=bias_type, train_dfs=split_train_dfs,
                                                  eval_dfs=split_eval_dfs, pearson_corr=pearson_corr,
                                                  spearman_corr=spearman_corr, mae=mean_absolute_error)

                model = MonoTransQuestModel(MODEL_TYPE, monotransquest_config["best_model_dir"], num_labels=1,
                                            use_cuda=torch.cuda.is_available(), args=monotransquest_config, **kwargs)
                result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
                                                                            spearman_corr=spearman_corr, rmse=rmse,
                                                                            mae=mean_absolute_error,
                                                                            partial_input="both")
                if bias_model_type == "partial_input_target":
                    result_target, model_outputs_target, wrong_predictions_target = model.eval_model(dev_bias_model,
                                                                                                     pearson_corr=pearson_corr,
                                                                                                     spearman_corr=spearman_corr,
                                                                                                     rmse=rmse,
                                                                                                     mae=mean_absolute_error,
                                                                                                     partial_input="target")
                if bias_model_type == "partial_input_source":
                    result_target, model_outputs_target, wrong_predictions_target = model.eval_model(dev_bias_model,
                                                                                                     pearson_corr=pearson_corr,
                                                                                                     spearman_corr=spearman_corr,
                                                                                                     rmse=rmse,
                                                                                                     mae=mean_absolute_error,
                                                                                                     partial_input="source")
                predictions, raw_outputs = model.predict(test_sentence_pairs)
                dev_preds[:, i] = model_outputs
                test_preds[:, i] = predictions

            dev['predictions'] = dev_preds.mean(axis=1)
            test['predictions'] = test_preds.mean(axis=1)

        else:
            model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                                        args=monotransquest_config, **kwargs)

            if bias_model_type == "partial_input_target" or bias_model_type == "partial_input_source":
                bias_model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1,
                                                 use_cuda=torch.cuda.is_available(),
                                                 args=monotransquest_config, **kwargs)
                bias_type = 'partial_input'
            elif bias_model_type == 'sentence_length':
                bias_model = SentenceLengthBiasNet()
                bias_type = 'sentence_length'
            else:
                print('Please specify the bias model type.')

            # Get the train and eval dataset
            split_train_dfs = []
            split_eval_dfs = []
            for train_df in train_dfs:
                train_df, eval_df = train_test_split(train_df, test_size=0.1, random_state=SEED)
                split_train_dfs.append(train_df)
                split_eval_dfs.append(eval_df)

            model.train_model_with_focal_loss(bias_model=bias_model, bias_type=bias_type, train_dfs=split_train_dfs,
                                              eval_dfs=split_eval_dfs, pearson_corr=pearson_corr,
                                              spearman_corr=spearman_corr,
                                              mae=mean_absolute_error)

            model = MonoTransQuestModel(MODEL_TYPE, monotransquest_config["best_model_dir"], num_labels=1,
                                        use_cuda=torch.cuda.is_available(), args=monotransquest_config, **kwargs)
            result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
                                                                        spearman_corr=spearman_corr, rmse=rmse,
                                                                        mae=mean_absolute_error, partial_input="both")
            if bias_model_type == "partial_input_target":
                result_target, model_outputs_target, wrong_predictions_target = model.eval_model(dev_bias_model,
                                                                                                 pearson_corr=pearson_corr,
                                                                                                 spearman_corr=spearman_corr,
                                                                                                 rmse=rmse,
                                                                                                 mae=mean_absolute_error,
                                                                                                 partial_input="target")

            if bias_model_type == "partial_input_source":
                result_target, model_outputs_target, wrong_predictions_target = model.eval_model(dev_bias_model,
                                                                                                 pearson_corr=pearson_corr,
                                                                                                 spearman_corr=spearman_corr,
                                                                                                 rmse=rmse,
                                                                                                 mae=mean_absolute_error,
                                                                                                 partial_input="source")

            predictions, raw_outputs = model.predict(test_sentence_pairs)
            dev['predictions'] = model_outputs
            test['predictions'] = predictions



    else:
        model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                                    args=monotransquest_config, **kwargs)

        if bias_model_type == "partial_input_target" or bias_model_type == "partial_input_source":
            bias_model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                                             args=monotransquest_config, **kwargs)
            bias_type = 'partial_input'

        elif bias_model_type == 'sentence_length':
            bias_model = SentenceLengthBiasNet()
            bias_type = 'sentence_length'
        else:
            print('Please specify the bias model type.')

        model.train_model_with_focal_loss(bias_model=bias_model, bias_type=bias_type, train_dfs=train_dfs,
                                          pearson_corr=pearson_corr,
                                          spearman_corr=spearman_corr,
                                          mae=mean_absolute_error)
        result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
                                                                    spearman_corr=spearman_corr, rmse=rmse,
                                                                    mae=mean_absolute_error, partial_input="both")
        if bias_model_type == "partial_input_target":
            result_target, model_outputs_target, wrong_predictions_target = model.eval_model(dev_bias_model,
                                                                                             pearson_corr=pearson_corr,
                                                                                             spearman_corr=spearman_corr,
                                                                                             rmse=rmse,
                                                                                             mae=mean_absolute_error,
                                                                                             partial_input="target")

        if bias_model_type == "partial_input_source":
            result_target, model_outputs_target, wrong_predictions_target = model.eval_model(dev_bias_model,
                                                                                             pearson_corr=pearson_corr,
                                                                                             spearman_corr=spearman_corr,
                                                                                             rmse=rmse,
                                                                                             mae=mean_absolute_error,
                                                                                             partial_input="source")
        predictions, raw_outputs = model.predict(test_sentence_pairs)
        dev['predictions'] = model_outputs
        test['predictions'] = predictions

    dev = un_fit(dev, 'labels')
    dev = un_fit(dev, 'predictions')
    test = un_fit(test, 'predictions')
    dev.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
    draw_scatterplot(dev, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), language)
    print_stat(dev, 'labels', 'predictions')
    # format_submission(df=test, index=index, language_pair="en-de", method="TransQuest",
    #                 path=os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE))

    return dev, test


def train_with_partial_bias_model(language = "EN-DE", bias_model_type = "partial_input_target", label = "DA", is_sweeping = False, **kwargs):
    if is_sweeping:
        wandb_group = SWEEP_CONFIG['parameters']['wandb_group']['values'][0]
        with wandb.init(group = "Adv-partial-DA-EN-ZH") as run:

            sweep_config = wandb.config
            print('CONFIG (no defaults)',wandb.config)
            language = sweep_config['language']
            print(language)
            label = sweep_config['label']
            bias_model_type = sweep_config['bias_model_type']
            print(bias_model_type)

            config_defaults = monotransquest_config
            run.config.setdefaults(config_defaults)

            print(wandb_group)
            print('CONFIG',wandb.config)

            dev, test = train_MonoTransQuestFocalLoss(language, bias_model_type, label, sweep_config = sweep_config, **kwargs)

    else:

            dev, test = train_MonoTransQuestFocalLoss(language, bias_model_type, label, **kwargs)

    return dev, test


# Inference: Using the trained model for predictions

def predict_MonoTransQuest(language='EN-DE', model_language='EN-DE', dataset='test', evaluation_type="DA", data='MLQE',
                           task='regression', aug_type="shuffle", aux_type="focal", experiment=None, partial_input=None, shuffle_column=None, save_preds=True, **kwargs):

    if "task_config" in kwargs:
        task_config = kwargs.get("task_config")
        monotransquest_config.update(task_config)

    if data == 'wikimatrix':
        if language == 'EN-DE':
            file = "ende_9000_aug_custom_pipeline.tsv"
        else:
            print('Please specify WikiMatrix data file')
        train, dev, test = load_WikiMatrix_data(file=file, augmentation_type=aug_type, language=language, prep_for_training=True)

    else:
        if evaluation_type == 'DA':
            train, dev, test = load_MLQE_data(language=language, label='DA', prep_for_training=True)
        else:
            train, dev, test = load_MLQE_data(language=language, label='HTER', prep_for_training=True)

    if dataset == 'test':

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
    elif dataset == 'train':
        if partial_input == "target":
            train = train.drop('text_a', axis=1)
            train = train.rename(columns={"text_b": "text"})
            test_sentence_pairs = train['text'].to_list()

        elif partial_input == "source":
            train = train.drop('text_b', axis=1)
            train = train.rename(columns={"text_a": "text"})
            test_sentence_pairs = train['text'].to_list()

        else:
            if shuffle_column is not None:
                assert shuffle_column in ['text_a', 'text_b']
                train = swap_sentence_pairs(train, shuffle_column=shuffle_column)

            test_sentence_pairs = list(map(list, zip(train['text_a'].to_list(), train['text_b'].to_list())))
    else:
        print('Please specify the dataset used for predictions (either train, dev or test)')

    predictions = []

    model = MonoTransQuestModel(MODEL_TYPE, monotransquest_config["best_model_dir"], use_cuda=torch.cuda.is_available(),
                                args=monotransquest_config, **kwargs)

    predictions, raw_outputs = model.predict(test_sentence_pairs)
    # predictions.append(prediction)
    # print(index, ":", prediction)


    if save_preds:

        if experiment == "FINAL":
            seed = task_config['manual_seed']
            print("Seed:", seed)
            df_pred = pd.DataFrame(predictions, columns=['Predicted_' + evaluation_type])
            if language == model_language:
                if partial_input == "source":
                    df_pred.to_csv(
                        'DATA/MLQE-PE/' + model_language + '/predictions/' + aux_type + '/' + experiment + '/' + dataset + '_data/' + str(seed) + '_pred_'+ dataset + '_' + evaluation_type + '_mono_partial_source.csv')
                elif partial_input == "target":
                    df_pred.to_csv(
                        'DATA/MLQE-PE/' + model_language + '/predictions/' + aux_type + '/' + experiment + '/' + dataset + '_data/' + str(seed) + '_pred_' + dataset + '_' + evaluation_type + '_mono_partial_target.csv')
                elif shuffle_column == "text_a":
                    df_pred.to_csv(
                        'DATA/MLQE-PE/' + model_language + '/predictions/' + aux_type + '/' + experiment + '/' + dataset + '_data/' + str(seed) + '_pred_' + dataset + '_' + evaluation_type + '_mono_shuffle_source.csv')
                elif shuffle_column == "text_b":
                    df_pred.to_csv(
                        'DATA/MLQE-PE/' + model_language + '/predictions/' + aux_type + '/' + experiment + '/' + dataset + '_data/' + str(seed) + '_pred_' + dataset + '_' + evaluation_type + '_mono_shuffle_target.csv')

                else:
                    df_pred.to_csv(
                        'DATA/MLQE-PE/' + model_language + '/predictions/' + aux_type + '/' + experiment + '/' + dataset + '_data/' + str(seed) + '_pred_' + dataset + '_' + evaluation_type + '_mono.csv')

            # Out of domain predictions
            else: print("No OOD predictions for the FINAL folder")
        if data == 'wikimatrix':
            df_pred = pd.DataFrame(predictions, columns=['prediction'])
            if partial_input == "source":
                df_pred.to_csv(
                    'DATA/MLQE-PE/' + model_language + '/predictions/benchmark/wiki_data/pred_' + dataset +'_classification_shu_partial_source.csv')
            elif partial_input == "target":
                df_pred.to_csv(
                    'DATA/MLQE-PE/' + model_language + '/predictions/benchmark/wiki_data/pred_' + dataset +'_classification_shu_partial_target.csv')
            elif shuffle_column == "text_a":
                df_pred.to_csv(
                    'DATA/MLQE-PE/' + model_language + '/predictions/benchmark/wiki_data/pred_' + dataset +'_classification_shu_shuffle_source.csv')
            elif shuffle_column == "text_b":
                df_pred.to_csv(
                    'DATA/MLQE-PE/' + model_language + '/predictions/benchmark/wiki_data/pred_' + dataset +'_classification_shu_shuffle_target.csv')
            else:
                df_pred.to_csv(
                    'DATA/MLQE-PE/' + model_language + '/predictions/benchmark/wiki_data/pred_' + dataset +'_classification_shu.csv')

        else:
            df_pred = pd.DataFrame(predictions, columns=['Predicted_' + evaluation_type])
            if language == model_language:
                if partial_input == "source":
                    df_pred.to_csv(
                        'DATA/MLQE-PE/' + model_language + '/predictions/'+aux_type+'/'+experiment+'/' + dataset + '_data/pred_' + dataset + '_' + evaluation_type + '_mono_partial_source.csv')
                elif partial_input == "target":
                    df_pred.to_csv(
                        'DATA/MLQE-PE/' + model_language + '/predictions/'+aux_type+'/'+experiment+'/' + dataset + '_data/pred_' + dataset + '_' + evaluation_type + '_mono_partial_target.csv')
                elif shuffle_column == "text_a":
                    df_pred.to_csv(
                        'DATA/MLQE-PE/' + model_language + '/predictions/'+aux_type+'/'+experiment+'/' + dataset + '_data/pred_' + dataset + '_' + evaluation_type + '_mono_shuffle_source.csv')
                elif shuffle_column == "text_b":
                    df_pred.to_csv(
                        'DATA/MLQE-PE/' + model_language + '/predictions/'+aux_type+'/'+experiment+'/' + dataset + '_data/pred_' + dataset + '_' + evaluation_type + '_mono_shuffle_target.csv')

                else:
                    df_pred.to_csv(
                        'DATA/MLQE-PE/' + model_language + '/predictions/'+aux_type+'/'+experiment+'/' + dataset + '_data/pred_' + dataset + '_' + evaluation_type + '_mono.csv')

            # Out of domain predictions
            else:
                if partial_input == "source":
                    df_pred.to_csv(
                        'DATA/MLQE-PE/' + model_language + '/predictions/'+aux_type+'/'+experiment+'/' + dataset + '_data/OOD_' + language + '_pred_' + dataset + '_' + evaluation_type + '_mono_partial_source.csv')
                elif partial_input == "target":
                    df_pred.to_csv(
                        'DATA/MLQE-PE/' + model_language + '/predictions/'+aux_type+'/'+experiment+'/' + dataset + '_data/OOD_' + language + '_pred_' + dataset + '_' + evaluation_type + '_mono_partial_target.csv')
                elif shuffle_column == "text_a":
                    df_pred.to_csv(
                        'DATA/MLQE-PE/' + model_language + '/predictions/'+aux_type+'/'+experiment+'/' + dataset + '_data/OOD_' + language + '_pred_' + dataset + '_' + evaluation_type + '_mono_shuffle_source.csv')
                elif shuffle_column == "text_b":
                    df_pred.to_csv(
                        'DATA/MLQE-PE/' + model_language + '/predictions/'+aux_type+'/'+experiment+'/' + dataset + '_data/OOD_' + language + '_pred_' + dataset + '_' + evaluation_type + '_mono_shuffle_target.csv')

                else:
                    df_pred.to_csv(
                    'DATA/MLQE-PE/' + model_language + '/predictions/'+aux_type+'/'+experiment+'/' + dataset + '_data/OOD_' + language + '_pred_' + dataset + '_' + evaluation_type + '_mono.csv')

    return predictions




SWEEP_CONFIG = {
    'method': 'grid',
    'parameters': {
        'wandb_group': {
            'values': ['not-colab-test']
        },
        'language': {
            'values': ['EN-DE']
        },
        'label': {
            'values': ['DA']
        },

    }
}

def main():
    sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project="mono-transquest-focal-loss")
    wandb.agent(sweep_id, function=train_benchmark, count=20)
if __name__ == "__main__":
    main()