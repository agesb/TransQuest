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

multitransquest_config = {

    'output_dir': PROJECT_DIR + 'CODE/temp/outputs/',
    "best_model_dir": PROJECT_DIR + "CODE/temp/outputs/best_model/",
    'cache_dir': PROJECT_DIR + 'CODE/temp/cache_dir/',
    'data_dir': PROJECT_DIR + 'DATA/', #Can be used for additional analysis, is optional

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 80,
    'train_batch_size': 8, # 8 is the original
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 8,
    'num_train_epochs': 3, #3
    'weight_decay': 0,
    'learning_rate': 2e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False,

    'logging_steps': 300, #300
    'save_steps': 100000, #300
    "no_cache": False,
    'save_model_every_epoch': True,
    'n_fold': 1,#3
    'evaluate_during_training': True, #True
    'evaluate_during_training_steps': 300, #300
    "evaluate_during_training_verbose": True,
    'use_cached_eval_features': False,
    'save_eval_checkpoints': False,
    'tensorboard_dir': None,

    # Multi-task settings
    'num_tasks': 2,
    'regression': [True, True],
    'num_labels': [1, 1],
    'learning_rates': [2e-5, 2e-5],
    'adam_epsilons': [1e-8, 1e-8],
    'grad_weights': [1, 1],
    'task_weights': [0.5, 0.5],
    'task_weight_schedule': 'constant',
    'num_shared_decoder_layers': 0,
    'num_separate_decoder_layers': 1, # min 0, max 2

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    'silent': False,

    'wandb_project': None, #'multi-transquest',
    'wandb_kwargs': {},

    "use_early_stopping": False,
    "early_stopping_patience": 10,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,

    "manual_seed": 777,

    "encoding": None,
}
