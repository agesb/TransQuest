from CODE.transquest.training.run_mono import predict_MonoTransQuest, train_TransQuest
from CODE.transquest.training.util.prep_data import load_MLQE_data

def main():
    language = 'EN-DE' # Alternative: "EN-ZH" or any other language pair from the MLQE-PE dataset
    label = "DA" # Alternative: "HTER"
    val_set = "test" # Alternative: "dev"
    # Training
    train_TransQuest(language = language, data = 'MLQE', label = label, focal_loss=True)
    # Predict on both sentences
    predictions = predict_MonoTransQuest(language=language, model_language=language, dataset=val_set, aux_type="focal",
                                experiment = "demo", evaluation_type=label, data='MLQE', task='regression', save_preds=False)
    # Predict on partial input (target)
    predictions_on_target = predict_MonoTransQuest(language=language, model_language=language, dataset=val_set, aux_type="focal",
                                experiment = "demo", evaluation_type=label, data='MLQE', task='regression', save_preds=False, partial_input="target")

    # Get true labels
    _, original_dev, original_test = load_MLQE_data(language, label)
    if val_set == 'dev':
        original_dataframe = original_dev
    else:
        original_dataframe = original_test

    # Output
    original_dataframe['preds'] = predictions
    if label == "HTER":
        print("\nPearson correlation of the HTER predictions and labels (test set): \t", original_dataframe['HTER'].corr(original_dataframe['preds']))
    else:
        print("\nPearson correlation of the DA predictions and labels (test set): \t", original_dataframe['z_mean'].corr(original_dataframe['preds']))

if __name__ == "__main__":
    main()