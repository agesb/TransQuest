import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
import torch
from nlp import Dataset


class PreTrainedDataset(Dataset):

    def __init__(self, tokenizer, df, combination='both', evaluation_type='DA'):
        assert combination in ['both', 'original', 'translation']
        assert evaluation_type in ['HTER', 'DA']

        self.tokenizer = tokenizer
        if evaluation_type == 'DA':
            self.labels = df['z_mean']
        else:
            self.labels = df['HTER']

        self.combination = combination

        self.input = list(zip(df['original'], df['translation']))

    def collate_fn(self, batch):
        batch_features_original = [f[0] for f, l in batch]
        batch_features_translation = [f[1] for f, l in batch]
        batch_labels = [l for f, l in batch]

        batch_labels = torch.FloatTensor(batch_labels)

        if self.combination == 'both':
            batch_inputs = self.tokenizer(batch_features_original, batch_features_translation, return_tensors="pt")
        elif self.combination == 'original':
            batch_inputs = self.tokenizer(batch_features_original, return_tensors="pt")
        if self.combination == 'translation':
            batch_inputs = self.tokenizer(batch_features_translation, return_tensors="pt")

        return batch_inputs, batch_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.input[item], self.labels[item]


def predict(model, data_loader, device):
    model.to(device)
    model.eval()
    logits_all = []

    with torch.no_grad():
        for batch in data_loader:
            feature, target = batch

            target = target.to(device)
            feature = feature.to(device)
            for k, v in feature.items():
                feature[k] = feature[k].to(device)

            outputs = model(**feature, labels=target)
            logits = outputs.logits.squeeze(1).detach().cpu().numpy()

            logits_all.extend(logits)

    return logits_all


def run_TransQuestHF(dataframe, device, language='EN-DE', model_version='monotransquest-da-en_de-wiki', evaluation_type='DA',
                   combination='both'):
    assert language in ['EN-DE', 'EN-ZH', 'ET-EN', 'RO-EN', 'SI-EN', 'NE-EN', 'RU-EN']

    # Initialise the Huggingface TransQuest tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TransQuest/" + model_version)

    # Initialise the pre-trained Huggingface TransQuest model
    model = AutoModelForSequenceClassification.from_pretrained("TransQuest/" + model_version)
    model.to(device)

    # Tokenize the dataset and create a data loader
    tokenized_dataset = PreTrainedDataset(tokenizer, dataframe, combination=combination,
                                          evaluation_type=evaluation_type)
    data_loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=1, collate_fn=tokenized_dataset.collate_fn)

    # Get predictions
    predictions = predict(model, data_loader)

    # Save predictions as CSV
    df = pd.DataFrame(predictions, columns=['Predicted_' + evaluation_type])
    df.to_csv(
        'DATA/EN-DE/predictions/pred_OOD_ET' + model_version + '_' + evaluation_type + '_' + combination + '.csv')
    # Save histogram of score distribution
    if evaluation_type == 'DA':
        plt.hist(dataframe['z_mean'], label="Mean DA Score Distribution", color=(1, 1, 1, 0), edgecolor="r", bins=20)
        pearson_R = str(dataframe['z_mean'].corr(pd.Series(predictions)))
    else:
        plt.hist(dataframe['HTER'], label="Mean HTER Score Distribution", color=(1, 1, 1, 0), edgecolor="r", bins=20)
        pearson_R = str(dataframe['HTER'].corr(pd.Series(predictions)))
    plt.hist(predictions, label="TransQuest predicted distribution", color=(0, 0, 1, 0.5), bins=20)
    plt.legend()

    # ADD CORRELATION TO TITLE
    plt.title(
        evaluation_type + " EN-DE large model OOD on " + language + "\nPearson R (real & predicted scores): " + pearson_R)
    plt.xlabel(evaluation_type + " Score")
    # plt.savefig('drive/MyDrive/Adversarial_MTQE/DATA/'+language+'/predictions/'+model_version+'.png')
    plt.show()

    return df