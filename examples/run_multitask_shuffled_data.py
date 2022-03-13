from CODE.transquest.training.run_multi import predict_MultiTransQuest, multitask_augmented_wiki_data
from CODE.transquest.training.util.prep_data import load_MLQE_data

def main():
    language = 'EN-DE' # Alternative: "EN-ZH" or any other language pair from the MLQE-PE dataset
    label = "DA" # Alternative: "HTER"
    val_set = "test" # Alternative: "dev"
    # Task specific configurations
    TASK_CONFIG = {
        'regression': [True, False],
        'num_labels': [1, 2],
        'manual_seed': 777,
        'num_shared_decoder_layers': 1,
        'num_separate_decoder_layers': 0,
        'learning_rates': [2e-5, 2e-5],
        'train_batch_size': 16
    }
    # Training
    multitask_augmented_wiki_data(language = language, label= label, is_sweeping=False, task_config=TASK_CONFIG)
    # Predict on both sentences
    predictions, _ = predict_MultiTransQuest(language=language, model_language=language, dataset=val_set, save_results=False,
                            aux_type="shuffle", experiment="demo", evaluation_type=label, data='MLQE', task='regression',
                            task_config=TASK_CONFIG)
    # Predict on partial input (target)
    predictions_on_target, _ = predict_MultiTransQuest(language=language, model_language=language, dataset=val_set, save_results=False,
                            aux_type="shuffle", experiment = "demo", evaluation_type=label, data='MLQE', task='regression',
                            partial_input="target", task_config=TASK_CONFIG)
    # Get true labels
    _, original_dev, original_test = load_MLQE_data(language, label)
    if val_set == 'dev':
        original_dataframe = original_dev
    else:
        original_dataframe = original_test
    # Output
    if label == "HTER":
        print("\nPearson correlation of the HTER predictions and labels (test set): \t", original_dataframe['HTER'].corr(predictions['Predicted_HTER']))
    else:
        print("\nPearson correlation of the DA predictions and labels (test set): \t", original_dataframe['z_mean'].corr(predictions['Predicted_DA']))

if __name__ == "__main__":
    main()