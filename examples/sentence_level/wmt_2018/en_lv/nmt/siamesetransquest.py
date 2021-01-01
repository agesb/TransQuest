import csv
import logging
import math
import os
import shutil

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from examples.sentence_level.wmt_2018.common.util.download import download_from_google_drive
from examples.sentence_level.wmt_2018.common.util.draw import draw_scatterplot, print_stat
from examples.sentence_level.wmt_2018.common.util.normalizer import fit, un_fit
from examples.sentence_level.wmt_2018.common.util.postprocess import format_submission
from examples.sentence_level.wmt_2018.common.util.reader import read_annotated_file, read_test_file
from examples.sentence_level.wmt_2018.en_lv.nmt.siamesetransquest_config import TEMP_DIRECTORY, GOOGLE_DRIVE, DRIVE_FILE_ID, \
    MODEL_NAME, siamesetransquest_config, SEED, RESULT_FILE, SUBMISSION_FILE, RESULT_IMAGE
from transquest.algo.sentence_level.siamesetransquest import LoggingHandler, SentencesDataset, \
    SiameseTransQuestModel
from transquest.algo.sentence_level.siamesetransquest import models, losses
from transquest.algo.sentence_level.siamesetransquest.evaluation import EmbeddingSimilarityEvaluator
from transquest.algo.sentence_level.siamesetransquest.readers import QEDataReader

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

if GOOGLE_DRIVE:
    download_from_google_drive(DRIVE_FILE_ID, MODEL_NAME)

TRAIN_FOLDER = "examples/sentence_level/wmt_2018/en_lv/data/en_lv"
DEV_FOLDER = "examples/sentence_level/wmt_2018/en_lv/data/en_lv"
TEST_FOLDER = "examples/sentence_level/mt_2018/en_lv/data/en_lv"

train = read_annotated_file(path=TRAIN_FOLDER, original_file="train.nmt.src", translation_file="train.nmt.mt", hter_file="train.nmt.hter")
dev = read_annotated_file(path=DEV_FOLDER, original_file="dev.nmt.src", translation_file="dev.nmt.mt", hter_file="dev.nmt.hter")
test = read_test_file(path=TEST_FOLDER, original_file="test.nmt.src", translation_file="test.nmt.mt")

index = test['index'].to_list()

train = train[['original', 'translation', 'hter']]
dev = dev[['original', 'translation', 'hter']]
test = test[['original', 'translation']]

train = train.rename(columns={'original': 'text_a', 'translation': 'text_b', 'hter': 'labels'}).dropna()
dev = dev.rename(columns={'original': 'text_a', 'translation': 'text_b', 'hter': 'labels'}).dropna()
test = test.rename(columns={'original': 'text_a', 'translation': 'text_b'}).dropna()

train = fit(train, 'labels')
dev = fit(dev, 'labels')


if siamesetransquest_config["evaluate_during_training"]:
    if siamesetransquest_config["n_fold"] > 0:
        dev_preds = np.zeros((len(dev), siamesetransquest_config["n_fold"]))
        test_preds = np.zeros((len(test), siamesetransquest_config["n_fold"]))
        for i in range(siamesetransquest_config["n_fold"]):

            if os.path.exists(siamesetransquest_config['best_model_dir']) and os.path.isdir(
                    siamesetransquest_config['best_model_dir']):
                shutil.rmtree(siamesetransquest_config['best_model_dir'])

            if os.path.exists(siamesetransquest_config['cache_dir']) and os.path.isdir(
                    siamesetransquest_config['cache_dir']):
                shutil.rmtree(siamesetransquest_config['cache_dir'])

            os.makedirs(siamesetransquest_config['cache_dir'])

            train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
            train_df.to_csv(os.path.join(siamesetransquest_config['cache_dir'], "train.tsv"), header=True, sep='\t',
                            index=False, quoting=csv.QUOTE_NONE)
            eval_df.to_csv(os.path.join(siamesetransquest_config['cache_dir'], "eval_df.tsv"), header=True, sep='\t',
                           index=False, quoting=csv.QUOTE_NONE)
            dev.to_csv(os.path.join(siamesetransquest_config['cache_dir'], "dev.tsv"), header=True, sep='\t',
                       index=False, quoting=csv.QUOTE_NONE)
            test.to_csv(os.path.join(siamesetransquest_config['cache_dir'], "test.tsv"), header=True, sep='\t',
                        index=False, quoting=csv.QUOTE_NONE)

            sts_reader = QEDataReader(siamesetransquest_config['cache_dir'], s1_col_idx=0, s2_col_idx=1,
                                      score_col_idx=2,
                                      normalize_scores=False, min_score=0, max_score=1, header=True)

            word_embedding_model = models.Transformer(MODEL_NAME, max_seq_length=siamesetransquest_config[
                'max_seq_length'])

            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True,
                                           pooling_mode_cls_token=False,
                                           pooling_mode_max_tokens=False)

            model = SiameseTransQuestModel(modules=[word_embedding_model, pooling_model])
            train_data = SentencesDataset(sts_reader.get_examples('train.tsv'), model)
            train_dataloader = DataLoader(train_data, shuffle=True,
                                          batch_size=siamesetransquest_config['train_batch_size'])
            train_loss = losses.CosineSimilarityLoss(model=model)

            eval_data = SentencesDataset(examples=sts_reader.get_examples('eval_df.tsv'), model=model)
            eval_dataloader = DataLoader(eval_data, shuffle=False,
                                         batch_size=siamesetransquest_config['train_batch_size'])
            evaluator = EmbeddingSimilarityEvaluator(eval_dataloader)

            warmup_steps = math.ceil(
                len(train_data) * siamesetransquest_config["num_train_epochs"] / siamesetransquest_config[
                    'train_batch_size'] * 0.1)

            model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=evaluator,
                      epochs=siamesetransquest_config['num_train_epochs'],
                      evaluation_steps=100,
                      optimizer_params={'lr': siamesetransquest_config["learning_rate"],
                                        'eps': siamesetransquest_config["adam_epsilon"],
                                        'correct_bias': False},
                      warmup_steps=warmup_steps,
                      output_path=siamesetransquest_config['best_model_dir'])

            model = SiameseTransQuestModel(siamesetransquest_config['best_model_dir'])

            dev_data = SentencesDataset(examples=sts_reader.get_examples("dev.tsv"), model=model)
            dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=8)
            evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
            model.evaluate(evaluator,
                           result_path=os.path.join(siamesetransquest_config['cache_dir'], "dev_result.txt"))

            test_data = SentencesDataset(examples=sts_reader.get_examples("test.tsv", test_file=True), model=model)
            test_dataloader = DataLoader(test_data, shuffle=False, batch_size=8)
            evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
            model.evaluate(evaluator,
                           result_path=os.path.join(siamesetransquest_config['cache_dir'], "test_result.txt"),
                           verbose=False)

            with open(os.path.join(siamesetransquest_config['cache_dir'], "dev_result.txt")) as f:
                dev_preds[:, i] = list(map(float, f.read().splitlines()))

            with open(os.path.join(siamesetransquest_config['cache_dir'], "test_result.txt")) as f:
                test_preds[:, i] = list(map(float, f.read().splitlines()))

        dev['predictions'] = dev_preds.mean(axis=1)
        test['predictions'] = test_preds.mean(axis=1)

dev = un_fit(dev, 'labels')
dev = un_fit(dev, 'predictions')
test = un_fit(test, 'predictions')
dev.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
draw_scatterplot(dev, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), "English-Latvian-NMT")
print_stat(dev, 'labels', 'predictions')
format_submission(df=test, index=index, method="SiameseTransQuest",
                  path=os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE))