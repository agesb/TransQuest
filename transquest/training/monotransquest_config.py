from multiprocessing import cpu_count

SEED = 777

PROJECT_DIR = "/vol/bitbucket/hsb20/msc-project-hanna-behnke/"
TEMP_DIRECTORY = PROJECT_DIR + "CODE/temp/data"


RESULT_FILE = "result.tsv"
SUBMISSION_FILE = "predictions.txt"
RESULT_IMAGE = "result.jpg"
GOOGLE_DRIVE = False
DRIVE_FILE_ID = None
MODEL_TYPE = "xlmroberta"
MODEL_NAME = "xlm-roberta-base"

monotransquest_config = {
    'output_dir': PROJECT_DIR + 'CODE/temp/outputs/',
    "best_model_dir": PROJECT_DIR + "CODE/temp/outputs/best_model/",
    'cache_dir': PROJECT_DIR + 'CODE/temp/cache_dir/',
    'data_dir': PROJECT_DIR + 'DATA/', #Can be used for additional analysis, is optional

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 80,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 8,
    'num_train_epochs': 3,#3
    'weight_decay': 0,
    'learning_rate': 2e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False,

    'logging_steps': 300,
    'save_steps': 100000,
    "no_cache": False,
    'save_model_every_epoch': False,
    'n_fold': 1,#3
    'evaluate_during_training': False,
    'evaluate_during_training_steps': 300,
    "evaluate_during_training_verbose": False,
    'use_cached_eval_features': False,
    'save_eval_checkpoints': False,
    'tensorboard_dir': None,

    'regression': True,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    'silent': False,

    # TODO set none before handing in the code
    'wandb_project': None, #'multi-transquest'
    'wandb_kwargs': None, #{'entity': 'username'}

    # Settings for focal loss training:
    'bias_model': "partial_input_target", # alternatively, put "partial_input_source" or "sentence_length"
    'bias_model_weight': None,
    'lr_sentence_length': 0.01, # For focal loss with sentence length bias model, only


    "use_early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,

    "manual_seed": 777,

    "encoding": None,
}
