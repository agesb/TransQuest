#!/usr/bin/env python
# coding: utf-8



from __future__ import absolute_import, division, print_function

import glob
import logging
import math
import os
import sys
import random
import shutil
import tempfile
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data.dataset import ConcatDataset
from scipy.stats import mode
from sklearn.metrics import (
    confusion_matrix,
    label_ranking_average_precision_score,
    matthews_corrcoef,
)
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import tqdm, trange
from transformers import (
    BertConfig,
    BertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    LongformerConfig,
    LongformerForSequenceClassification,
    LongformerTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    XLMConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMTokenizer,
)
from transformers.convert_graph_to_onnx import convert, quantize
from transformers.optimization import AdamW, Adafactor
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)



# COLAB
PROJECT_PATH='drive/MyDrive/Adversarial_MTQE/'

# LAB MACHINE
#PROJECT_PATH='/vol/bitbucket/hsb20/'

sys.path.append(PROJECT_PATH+'CODE/transquest/algo/sentence_level/multitransquest/models')
sys.path.append(PROJECT_PATH+'CODE/transquest/algo/sentence_level/multitransquest')

# TODO Delete these imports as well as the corresponding code in models
#from bert_model import BertForSequenceClassification
#from distilbert_model import DistilBertForSequenceClassification
#from xlm_model import XLMForSequenceClassification

from load_data import MultitaskSampler
from roberta_model import RobertaForMultitaskSequenceClassification
from xlm_roberta_model import XLMRobertaForMultitaskSequenceClassification
from model_args import MultiTransQuestArgs
from utils import LazyClassificationDataset, InputExample, \
    convert_examples_to_features, sweep_config_to_sweep_values
from evaluation import pearson_corr, spearman_corr, rmse


'''
from transquest.algo.sentence_level.multitransquest.model_args import MultiTransQuestArgs
from transquest.algo.sentence_level.multitransquest.models.bert_model import BertForSequenceClassification
from transquest.algo.sentence_level.multitransquest.models.distilbert_model import DistilBertForSequenceClassification
from transquest.algo.sentence_level.multitransquest.models.roberta_model import RobertaForMultitaskSequenceClassification
from transquest.algo.sentence_level.multitransquest.models.xlm_model import XLMForSequenceClassification
from transquest.algo.sentence_level.multitransquest.models.xlm_roberta_model import XLMRobertaMultitaskForSequenceClassification
from transquest.algo.sentence_level.multitransquest.utils import LazyClassificationDataset, InputExample, \
    convert_examples_to_features
from transquest.algo.sentence_level.multitransquest.utils import sweep_config_to_sweep_values
'''

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)


class MultiTransQuestModel:
    def __init__(
            self,
            model_type,
            model_name,
            wandb_group,
            num_tasks = None,
            weight=None,
            args=None,
            use_cuda=True,
            cuda_device=-1,
            onnx_execution_provider=None,
            **kwargs,
    ):

        """
        Initializes a MonoTransQuest model.

        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            num_tasks: The number of tasks used (this may only include regression and classification tasks).
            weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            onnx_execution_provider (optional): ExecutionProvider to use with ONNX Runtime. Will use CUDA (if use_cuda) or CPU (if use_cuda is False) by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"


        MODEL_CLASSES = {

            # "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
            # "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
            # "longformer": (LongformerConfig, LongformerForSequenceClassification, LongformerTokenizer),
            "roberta": (RobertaConfig, RobertaForMultitaskSequenceClassification, RobertaTokenizer),
            # "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
            "xlmroberta": (XLMRobertaConfig, XLMRobertaForMultitaskSequenceClassification, XLMRobertaTokenizer),
        }

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, MultiTransQuestArgs):
            self.args = args

        # Store the number of tasks
        if num_tasks:
            self.num_tasks = num_tasks
        else:
            self.num_tasks = self.args.num_tasks

        self.wandb_group = wandb_group

        if self.args.thread_count:
            torch.set_num_threads(self.args.thread_count)

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

        self.config = config_class.from_pretrained(model_name, **self.args.config)
        # Add the multitask-specific settings
        self.config.update({'num_tasks': self.args.num_tasks, 'num_shared_decoder_layers':self.args.num_shared_decoder_layers, \
                            'list_labels': self.args.num_labels, 'grad_weights': self.args.grad_weights, 'id2label':None, 'label2id':None})

        # num_labels is a list of num_labels per task
        self.num_labels_per_task = self.args.num_labels
        self.weight = weight

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        if not self.args.quantized_model:
            if self.weight:
                self.model = model_class.from_pretrained(
                    model_name, config=self.config, weight=torch.Tensor(self.weight).to(self.device), **kwargs,
                )
            else:
                self.model = model_class.from_pretrained(model_name, config=self.config, **kwargs)
        else:
            quantized_weights = torch.load(os.path.join(model_name, "pytorch_model.bin"))
            if self.weight:
                self.model = model_class.from_pretrained(
                    None,
                    config=self.config,
                    state_dict=quantized_weights,
                    weight=torch.Tensor(self.weight).to(self.device),
                )
            else:
                self.model = model_class.from_pretrained(None, config=self.config, state_dict=quantized_weights)

        if self.args.dynamic_quantize:
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        if self.args.quantized_model:
            self.model.load_state_dict(quantized_weights)
        if self.args.dynamic_quantize:
            self.args.quantized_model = True

        self.results = [{} for i in range(self.num_tasks)]

        if not use_cuda:
            self.args.fp16 = False

        if self.args.fp16:
            try:
                from torch.cuda import amp
            except AttributeError:
                raise AttributeError("fp16 requires Pytorch >= 1.6. Please update Pytorch or turn off fp16.")

        if model_name in [
            "vinai/bertweet-base",
            "vinai/bertweet-covid19-base-cased",
            "vinai/bertweet-covid19-base-uncased",
        ]:
            self.tokenizer = tokenizer_class.from_pretrained(
                model_name, do_lower_case=self.args.do_lower_case, normalization=True, **kwargs
            )
        else:
            self.tokenizer = tokenizer_class.from_pretrained(
                model_name, do_lower_case=self.args.do_lower_case, **kwargs
            )

        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(self.args.special_tokens_list, special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.args.model_name = model_name
        self.args.model_type = model_type

        if model_type in ["camembert", "xlmroberta"]:
            warnings.warn(
                f"use_multiprocessing automatically disabled as {model_type}"
                " fails when using multiprocessing for feature conversion."
            )
            self.args.use_multiprocessing = False

        if self.args.wandb_project and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args.wandb_project = None

     def train_model(
            self,
            train_df_list,
            multi_label=False,
            output_dir=None,
            show_running_loss=True,
            args=None,
            eval_df=None,
            verbose=True,
            **kwargs,
    ):
        """
        Trains the model using 'train_df_list' – A list of the dataframes used for the different tasks

        Args:
            train_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be trained on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_df (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        # Args should include information in the number of tasks, training mode per task, number of labels per task and weighting of the tasks.
        if args:
            self.args.update_from_dict(args)

            # Assert that the information provided for the tasks corresponds to the number of tasks
            assert len(self.args.regression) == self.args.num_tasks
            assert len(self.args.num_labels) == self.args.num_tasks
            assert len(self.args.adversarial_loss) == self.args.num_tasks

        if self.args.silent:
            show_running_loss = False

        if self.args.evaluate_during_training and eval_df is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_df is not specified."
                " Pass eval_df to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set overwrite_output_dir: True to automatically overwrite.".format(output_dir)
            )
        self._move_model_to_device()

        # List to store the train datasets
        train_dataset_list = []

        # Iterate through all dataframes to create PyTorch Datasets
        for index, train_df in enumerate(train_df_list):
            if isinstance(train_df, str) and self.args.lazy_loading:
                if self.args.sliding_window:
                    raise ValueError("Lazy loading cannot be used with sliding window.")
                if self.args.model_type == "layoutlm":
                    raise NotImplementedError("Lazy loading is not implemented for LayoutLM models")
                train_dataset = LazyClassificationDataset(train_df, self.tokenizer, self.args)
            else:
                if self.args.lazy_loading:
                    raise ValueError("Input must be given as a path to a file when using lazy loading")
                if "text" in train_df.columns and "labels" in train_df.columns:
                    '''
                    # Disable layoutlm for now
                    if self.args.model_type == "layoutlm":
                        train_examples = [
                            InputExample(i, text, None, label, x0, y0, x1, y1)
                            for i, (text, label, x0, y0, x1, y1) in enumerate(
                                zip(
                                    train_df["text"].astype(str),
                                    train_df["labels"],
                                    train_df["x0"],
                                    train_df["y0"],
                                    train_df["x1"],
                                    train_df["y1"],
                                )
                            )
                        ]
                    else:
                    '''
                    train_examples = [
                        InputExample(i, text, None, label)
                        for i, (text, label) in enumerate(zip(train_df["text"].astype(str), train_df["labels"]))
                    ]
                elif "text_a" in train_df.columns and "text_b" in train_df.columns:
                    '''
                    if self.args.model_type == "layoutlm":
                        raise ValueError("LayoutLM cannot be used with sentence-pair tasks")
                    else:
                    '''
                    train_examples = [
                        InputExample(i, text_a, text_b, label)
                        for i, (text_a, text_b, label) in enumerate(
                            zip(train_df["text_a"].astype(str), train_df["text_b"].astype(str), train_df["labels"])
                        )
                    ]
                else:
                    warnings.warn(
                        "Dataframe headers not specified. Falling back to using column 0 as text and column 1 as labels."
                    )
                    train_examples = [
                        InputExample(i, text, None, label)
                        for i, (text, label) in enumerate(zip(train_df.iloc[:, 0], train_df.iloc[:, 1]))
                    ]
                # Load_and_cache_examples gets information on the current task to select the matching parameters from the config file
                train_dataset = self.load_and_cache_examples(train_examples, curr_task=index, verbose=verbose)
                train_dataset_list.append(train_dataset)


        #Concatente the datasets and load them with the MultitaskSampler to get one batch per task
        train_dataset_concat = ConcatDataset(train_dataset_list)
        train_sampler = MultitaskSampler(dataset = train_dataset_concat, batch_size = self.args.train_batch_size)
        train_dataloader = DataLoader(
            train_dataset_concat,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.train(
            train_dataloader,
            output_dir,
            multi_label=multi_label,
            show_running_loss=show_running_loss,
            eval_df=eval_df,
            verbose=verbose,
            **kwargs,
        )

        # model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        # model_to_save.save_pretrained(output_dir)
        # self.tokenizer.save_pretrained(output_dir)
        # torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        self.save_model(model=self.model)

        if verbose:
            logger.info(" Training of {} model complete. Saved to {}.".format(self.args.model_type, output_dir))

        return global_step, training_details

    def train(
            self,
            train_dataloader,
            output_dir,
            multi_label=False,
            show_running_loss=True,
            eval_df=None,
            verbose=True,
            **kwargs,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args

        tb_writer = SummaryWriter(logdir=args.tensorboard_dir)

        # len(dataloader) = number of batches
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [p for n, p in model.named_parameters() if n in params]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps

        if args.optimizer == "AdamW":
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        elif args.optimizer == "Adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adafactor_eps,
                clip_threshold=args.adafactor_clip_threshold,
                decay_rate=args.adafactor_decay_rate,
                beta1=args.adafactor_beta1,
                weight_decay=args.weight_decay,
                scale_parameter=args.adafactor_scale_parameter,
                relative_step=args.adafactor_relative_step,
                warmup_init=args.adafactor_warmup_init,
            )
            print("Using Adafactor for T5")
        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                    args.optimizer
                )
            )

        if args.scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)

        elif args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

        elif args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
            )

        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "polynomial_decay_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                lr_end=args.polynomial_decay_schedule_lr_end,
                power=args.polynomial_decay_schedule_lr_end,
            )

        else:
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0)
        epoch_number = 0
        best_eval_metric = [None] * self.num_tasks
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0
        current_loss = "Initializing"

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                        len(train_dataloader) // args.gradient_accumulation_steps
                )

                logger.info("   Continuing training from checkpoint, will skip to saved global_step")
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info("   Will skip the first %d steps in the current epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            # Create list of dictionaries to store the progress scores per task (metrics depend on the type of task)
            training_progress_scores = []
            for task in range(self.num_tasks):
                training_progress_scores_per_task = self._create_training_progress_scores(multi_label, curr_task = task, **kwargs)
                training_progress_scores.append(training_progress_scores_per_task)

        if args.wandb_project:
            if not wandb.setup().settings.sweep_id:
                logger.info(" Initializing WandB run for training.")
                wandb.init(project=args.wandb_project, config={**asdict(args)}, group=self.wandb_group, **args.wandb_kwargs)
            wandb.watch(self.model)

        if self.args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for _ in train_iterator:
            model.train()
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(f"Epoch {epoch_number + 1} of {args.num_train_epochs}")
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                # Compute the current task number
                curr_task = step % args.num_tasks

                inputs = self._get_inputs_dict(batch, curr_task)
                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        loss = outputs[0]
                else:
                    outputs = model(**inputs)
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    loss = outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if self.args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        scaler.unscale_(optimizer)
                    if args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if self.args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                    torch.cuda.empty_cache()

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss
                        if args.wandb_project or self.is_sweeping:
                            wandb.log(
                                {
                                    "training_loss_task_"+str(curr_task): current_loss,
                                    "lr": scheduler.get_last_lr()[0],
                                    "global_step": global_step,
                                }
                            )

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        if args.save_recent_only:
                            del_paths = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
                            for del_path in del_paths:
                                shutil.rmtree(del_path)
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        self.save_model(output_dir_current, optimizer, scheduler, model=model)

                    if args.evaluate_during_training and (
                            args.evaluate_during_training_steps > 0
                            and global_step % args.evaluate_during_training_steps == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        for eval_task in range(self.num_tasks):
                            results, _, _ = self.eval_model(
                                eval_df[eval_task], curr_task=eval_task,
                                verbose=verbose and args.evaluate_during_training_verbose,
                                silent=args.evaluate_during_training_silent,
                                wandb_log=False,
                                **kwargs,
                            )
                            for key, value in results.items():
                                tb_writer.add_scalar("eval_task_"+str(eval_task)+"{}".format(key), value, global_step)

                            output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                            if args.save_eval_checkpoints:
                                if args.save_recent_only:
                                    del_paths = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
                                    for del_path in del_paths:
                                        shutil.rmtree(del_path)
                                self.save_model(output_dir_current, optimizer, scheduler, model=model, results=results)

                            training_progress_scores[eval_task]["global_step"].append(global_step)
                            training_progress_scores[eval_task]["train_loss"].append(current_loss)
                            for key in results:
                                training_progress_scores[eval_task][key].append(results[key])
                            report = pd.DataFrame(training_progress_scores[eval_task])
                            report.to_csv(
                                os.path.join(args.output_dir, "training_progress_scores_task_"+str(eval_task)+".csv"), index=False,
                            )

                            if args.wandb_project or self.is_sweeping:
                                wandb.log(self._get_last_metrics(training_progress_scores[eval_task], eval_task))

                            if not best_eval_metric[eval_task]:
                                best_eval_metric[eval_task] = results[args.early_stopping_metric]
                                self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                            if best_eval_metric[eval_task] and args.early_stopping_metric_minimize:
                                if best_eval_metric[eval_task] - results[args.early_stopping_metric] > args.early_stopping_delta:
                                    best_eval_metric[eval_task] = results[args.early_stopping_metric]
                                    self.save_model(
                                        args.best_model_dir, optimizer, scheduler, model=model, results=results
                                    )
                                    early_stopping_counter = 0
                                else:
                                    if args.use_early_stopping:
                                        if early_stopping_counter < args.early_stopping_patience:
                                            early_stopping_counter += 1
                                            if verbose:
                                                logger.info(f" No improvement in {args.early_stopping_metric}")
                                                logger.info(f" Current step: {early_stopping_counter}")
                                                logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                        else:
                                            if verbose:
                                                logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                                logger.info(" Training terminated.")
                                                train_iterator.close()
                                            return (
                                                global_step,
                                                tr_loss / global_step
                                                if not self.args.evaluate_during_training
                                                else training_progress_scores,
                                            )
                            else:
                                if results[args.early_stopping_metric] - best_eval_metric[eval_task] > args.early_stopping_delta:
                                    best_eval_metric[eval_task] = results[args.early_stopping_metric]
                                    self.save_model(
                                        args.best_model_dir, optimizer, scheduler, model=model, results=results
                                    )
                                    early_stopping_counter = 0
                                else:
                                    if args.use_early_stopping:
                                        if early_stopping_counter < args.early_stopping_patience:
                                            early_stopping_counter += 1
                                            if verbose:
                                                logger.info(f" No improvement in {args.early_stopping_metric}")
                                                logger.info(f" Current step: {early_stopping_counter}")
                                                logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                        else:
                                            if verbose:
                                                logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                                logger.info(" Training terminated.")
                                                train_iterator.close()
                                            return (
                                                global_step,
                                                tr_loss / global_step
                                                if not self.args.evaluate_during_training
                                                else training_progress_scores[eval_task],
                                            )

            epoch_number += 1
            torch.cuda.empty_cache()
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            if args.save_model_every_epoch or args.evaluate_during_training:
                if args.save_recent_only:
                    del_paths = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
                    for del_path in del_paths:
                        shutil.rmtree(del_path)
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:

                for eval_task in range(self.num_tasks):

                    results, _, _ = self.eval_model(
                        eval_df[eval_task], curr_task = eval_task,
                        verbose=verbose and args.evaluate_during_training_verbose,
                        silent=args.evaluate_during_training_silent,
                        wandb_log=False,
                        **kwargs,
                    )

                    self.save_model(output_dir_current, optimizer, scheduler, results=results)

                    training_progress_scores[eval_task]["global_step"].append(global_step)
                    training_progress_scores[eval_task]["train_loss"].append(current_loss)
                    for key in results:
                        training_progress_scores[eval_task][key].append(results[key])
                    report = pd.DataFrame(training_progress_scores[eval_task])
                    report.to_csv(
                        os.path.join(args.output_dir, "training_progress_scores_task_" + str(eval_task) + ".csv"), index=False,
                    )


                    if args.wandb_project or self.is_sweeping:
                        wandb.log(self._get_last_metrics(training_progress_scores[eval_task], eval_task))

                    # TODO: Fix model save (this should not happen per task)
                    if not best_eval_metric[eval_task]:
                        best_eval_metric[eval_task] = results[args.early_stopping_metric]
                        self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                    if best_eval_metric[eval_task] and args.early_stopping_metric_minimize:
                        if best_eval_metric[eval_task] - results[args.early_stopping_metric] > args.early_stopping_delta:
                            best_eval_metric[eval_task] = results[args.early_stopping_metric]
                            self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                            early_stopping_counter = 0
                        else:
                            if args.use_early_stopping and args.early_stopping_consider_epochs:
                                if early_stopping_counter < args.early_stopping_patience:
                                    early_stopping_counter += 1
                                    if verbose:
                                        logger.info(f" No improvement in {args.early_stopping_metric}")
                                        logger.info(f" Current step: {early_stopping_counter}")
                                        logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                else:
                                    if verbose:
                                        logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                        logger.info(" Training terminated.")
                                        train_iterator.close()
                                    return (
                                        global_step,
                                        tr_loss / global_step
                                        if not self.args.evaluate_during_training
                                        else training_progress_scores,
                                    )
                    else:
                        if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                            best_eval_metric = results[args.early_stopping_metric]
                            self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                            early_stopping_counter = 0
                        else:
                            if args.use_early_stopping and args.early_stopping_consider_epochs:
                                if early_stopping_counter < args.early_stopping_patience:
                                    early_stopping_counter += 1
                                    if verbose:
                                        logger.info(f" No improvement in {args.early_stopping_metric}")
                                        logger.info(f" Current step: {early_stopping_counter}")
                                        logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                else:
                                    if verbose:
                                        logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                        logger.info(" Training terminated.")
                                        train_iterator.close()
                                    return (
                                        global_step,
                                        tr_loss / global_step
                                        if not self.args.evaluate_during_training
                                        else training_progress_scores[eval_task],
                                    )

        return (
            global_step,
            tr_loss / global_step if not self.args.evaluate_during_training else training_progress_scores,
        )

    def eval_model(
            self, eval_df, curr_task, multi_label=False, output_dir=None, verbose=True, silent=False, wandb_log=True, **kwargs
    ):
        """
        Evaluates the model on eval_df. Saves results to output_dir.

        Args:
            eval_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be evaluated on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            wandb_log: If True, evaluation results will be logged to wandb.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results.
            model_outputs: List of model outputs for each row in eval_df
            wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model
        """  # noqa: ignore flake8"

        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        result, model_outputs, wrong_preds = self.evaluate(
            eval_df, output_dir, curr_task, multi_label=multi_label, verbose=verbose, silent=silent, wandb_log=wandb_log, **kwargs
        )
        self.results[curr_task].update(result)

        if verbose:
            logger.info(self.results)

        return result, model_outputs, wrong_preds

    def evaluate(
            self, eval_df, output_dir, curr_task, multi_label=False, prefix="", verbose=True, silent=False, wandb_log=True,
            **kwargs
    ):
        """
        Evaluates the model on eval_df.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args
        eval_output_dir = output_dir

        results = {}
        if isinstance(eval_df, str) and self.args.lazy_loading:
            if self.args.model_type == "layoutlm":
                raise NotImplementedError("Lazy loading is not implemented for LayoutLM models")
            eval_dataset = LazyClassificationDataset(eval_df, self.tokenizer, self.args)
            eval_examples = None
        else:
            if self.args.lazy_loading:
                raise ValueError("Input must be given as a path to a file when using lazy loading")

            if "text" in eval_df.columns and "labels" in eval_df.columns:
                if self.args.model_type == "layoutlm":
                    eval_examples = [
                        InputExample(i, text, None, label, x0, y0, x1, y1)
                        for i, (text, label, x0, y0, x1, y1) in enumerate(
                            zip(
                                eval_df["text"].astype(str),
                                eval_df["labels"],
                                eval_df["x0"],
                                eval_df["y0"],
                                eval_df["x1"],
                                eval_df["y1"],
                            )
                        )
                    ]
                else:
                    eval_examples = [
                        InputExample(i, text, None, label)
                        for i, (text, label) in enumerate(zip(eval_df["text"].astype(str), eval_df["labels"]))
                    ]
            elif "text_a" in eval_df.columns and "text_b" in eval_df.columns:
                if self.args.model_type == "layoutlm":
                    raise ValueError("LayoutLM cannot be used with sentence-pair tasks")
                else:
                    eval_examples = [
                        InputExample(i, text_a, text_b, label)
                        for i, (text_a, text_b, label) in enumerate(
                            zip(eval_df["text_a"].astype(str), eval_df["text_b"].astype(str), eval_df["labels"])
                        )
                    ]
            else:
                warnings.warn(
                    "Dataframe headers not specified. Falling back to using column 0 as text and column 1 as labels."
                )
                eval_examples = [
                    InputExample(i, text, None, label)
                    for i, (text, label) in enumerate(zip(eval_df.iloc[:, 0], eval_df.iloc[:, 1]))
                ]

            if args.sliding_window:
                eval_dataset, window_counts = self.load_and_cache_examples(
                    eval_examples, curr_task=curr_task, evaluate=True, verbose=verbose, silent=silent
                )
            else:
                eval_dataset = self.load_and_cache_examples(
                    eval_examples, curr_task=curr_task, evaluate=True, verbose=verbose, silent=silent
                )
        os.makedirs(eval_output_dir, exist_ok=True)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(eval_dataloader)

        preds = np.empty((len(eval_dataset), self.num_labels_per_task[curr_task]))
        if multi_label:
            out_label_ids = np.empty((len(eval_dataset), self.num_labels_per_task[curr_task]))
        else:
            out_label_ids = np.empty((len(eval_dataset)))
        model.eval()

        if self.args.fp16:
            from torch.cuda import amp

        for i, batch in enumerate(tqdm(eval_dataloader, disable=args.silent or silent, desc="Running Evaluation")):
            # batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch, curr_task)

                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        tmp_eval_loss, logits = outputs[:2]
                else:
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                if multi_label:
                    logits = logits.sigmoid()

                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            start_index = self.args.eval_batch_size * i
            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else len(eval_dataset)
            preds[start_index:end_index] = logits.detach().cpu().numpy()

            out_label_ids[start_index:end_index] = inputs["labels"].detach().cpu().numpy()

            # if preds is None:
            #     preds = logits.detach().cpu().numpy()
            #     out_label_ids = inputs["labels"].detach().cpu().numpy()
            # else:
            #     preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            #     out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        if args.sliding_window:
            count = 0
            window_ranges = []
            for n_windows in window_counts:
                window_ranges.append([count, count + n_windows])
                count += n_windows

            preds = [preds[window_range[0]: window_range[1]] for window_range in window_ranges]
            out_label_ids = [
                out_label_ids[i] for i in range(len(out_label_ids)) if i in [window[0] for window in window_ranges]
            ]

            model_outputs = preds

            preds = [np.argmax(pred, axis=1) for pred in preds]
            final_preds = []
            for pred_row in preds:
                mode_pred, counts = mode(pred_row)
                if len(counts) > 1 and counts[0] == counts[1]:
                    final_preds.append(args.tie_value)
                else:
                    final_preds.append(mode_pred[0])
            preds = np.array(final_preds)
        elif not multi_label and args.regression[curr_task] is True:
            preds = np.squeeze(preds)
            model_outputs = preds
        else:
            model_outputs = preds

            if not multi_label:
                preds = np.argmax(preds, axis=1)

        result, wrong = self.compute_metrics(preds, curr_task, out_label_ids, eval_examples, **kwargs)
        result["eval_loss"] = eval_loss
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        # Weights and Biases for multiple tasks - regression
        if self.args.wandb_project and wandb_log and not multi_label and self.args.regression[curr_task]:
            if not wandb.setup().settings.sweep_id:
                logger.info(" Initializing WandB run for evaluation.")
                wandb.init(project=args.wandb_project, config={**asdict(args)}, group=self.wandb_group, **args.wandb_kwargs)

            if not self.args.sliding_window:
                # MSE
                wandb.log({"rmse_task_" + str(curr_task):result['rmse']})

                # Pearson R
                wandb.log({"corr_pearson_task_" + str(curr_task): result['pearson_corr']})

                # Spearman Correlation
                wandb.log({"corr_spearman_task_" + str(curr_task): result['spearman_corr']})

                regression_eval_table = wandb.Table(columns=["RMSE", "Pearson R", "Spearman"], data=[[result['rmse'], result['pearson_corr'], result['spearman_corr']]])
                wandb.log({"Regression Evaluation Task "+str(curr_task): regression_eval_table})

                plt.figure()

                plt.hist(eval_df["labels"], label="Mean Score Distribution (Task "+str(curr_task)+")", color=(1, 1, 1, 0), edgecolor="r")
                pearson_R = result['pearson_corr']
                plt.hist(preds, label="MultiTransQuest Predictions Distribution (Task "+str(curr_task)+")",
                         color=(0, 0, 1, 0.5))
                plt.legend()
                # ADD CORRELATION TO TITLE
                plt.title("Score Distribution Task"+str(curr_task)+"\nPearson R (real & predicted scores): " + str(pearson_R))
                plt.xlabel("Quality Score")

                plt.savefig("score_distribution_task_"+str(curr_task)+".jpg")
                img = Image.open("score_distribution_task_"+str(curr_task)+".jpg")
                wandb.log({"score_distribution_task"+str(curr_task): wandb.Image(img)})

        # Weights and Biases for multiple tasks - binary classification
        if self.args.wandb_project and wandb_log and not multi_label and not self.args.regression[curr_task]:
            if not wandb.setup().settings.sweep_id:
                logger.info(" Initializing WandB run for evaluation.")
                wandb.init(project=args.wandb_project,  config={**asdict(args)}, group=self.wandb_group, **args.wandb_kwargs)
            if not args.labels_map:
                self.args.labels_map = {i: i for i in range(self.num_labels_per_task[curr_task])}

            labels_list = sorted(list(self.args.labels_map.keys()))
            inverse_labels_map = {value: key for key, value in self.args.labels_map.items()}

            truth = [inverse_labels_map[out] for out in out_label_ids]

            # Confusion Matrix
            wandb.sklearn.plot_confusion_matrix(
                truth, [inverse_labels_map[pred] for pred in preds], labels=labels_list,
            )

            if not self.args.sliding_window:
                # ROC`
                wandb.log({"roc_task_"+str(curr_task): wandb.plots.ROC(truth, model_outputs, labels_list)})

                # Precision Recall
                wandb.log({"pr_task_"+str(curr_task): wandb.plots.precision_recall(truth, model_outputs, labels_list)})

        return results, model_outputs, wrong

    def load_and_cache_examples(
            self, examples, curr_task, evaluate=False, no_cache=False, multi_label=False, verbose=True, silent=False
    ):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        process_count = self.args.process_count

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        # Chose the output mode per task as specified in the mutlitransquest_config
        if not multi_label and args.regression[curr_task]:
            output_mode = "regression"
        else:
            output_mode = "classification"

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        # Chose the number of labels per task as specified in the multitransquest_config
        mode = "dev" if evaluate else "train"
        cached_features_file = os.path.join(
            args.cache_dir,
            "cached_{}_{}_{}_{}_{}".format(
                mode, args.model_type, args.max_seq_length, self.num_labels_per_task[curr_task], len(examples),
            ),
        )

        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not no_cache)
        ):
            features = torch.load(cached_features_file)
            if verbose:
                logger.info(f" Features loaded from cache at {cached_features_file}")
        else:
            if verbose:
                logger.info(" Converting to features started. Cache is not used.")
                if args.sliding_window:
                    logger.info(" Sliding window enabled")

            # If labels_map is defined, then labels need to be replaced with ints
            if self.args.labels_map and not self.args.regression[curr_task]:
                for example in examples:
                    if multi_label:
                        example.label = [self.args.labels_map[label] for label in example.label]
                    else:
                        example.label = self.args.labels_map[example.label]

            features = convert_examples_to_features(
                examples,
                args.max_seq_length,
                tokenizer,
                output_mode,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                # RoBERTa uses an extra separator b/w pairs of sentences,
                # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=bool(args.model_type in ["roberta", "camembert", "xlmroberta", "longformer"]),
                # PAD on the left for XLNet
                pad_on_left=bool(args.model_type in ["xlnet"]),
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                process_count=process_count,
                multi_label=multi_label,
                silent=args.silent or silent,
                use_multiprocessing=args.use_multiprocessing,
                sliding_window=args.sliding_window,
                flatten=not evaluate,
                stride=args.stride,
                add_prefix_space=bool(args.model_type in ["roberta", "camembert", "xlmroberta", "longformer"]),
                # avoid padding in case of single example/online inferencing to decrease execution time
                pad_to_max_length=bool(len(examples) > 1),
                args=args,
            )
            if verbose and args.sliding_window:
                logger.info(f" {len(features)} features created from {len(examples)} samples.")

            if not no_cache:
                torch.save(features, cached_features_file)

        if args.sliding_window and evaluate:
            features = [
                [feature_set] if not isinstance(feature_set, list) else feature_set for feature_set in features
            ]
            window_counts = [len(sample) for sample in features]
            features = [feature for feature_set in features for feature in feature_set]

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        if self.args.model_type == "layoutlm":
            all_bboxes = torch.tensor([f.bboxes for f in features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        if self.args.model_type == "layoutlm":
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_bboxes)
        else:
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        if args.sliding_window and evaluate:
            return dataset, window_counts
        else:
            return dataset

    def compute_metrics(self, preds, curr_task, labels, eval_examples=None, multi_label=False, **kwargs):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            preds: Model predictions
            labels: Ground truth labels
            eval_examples: List of examples on which evaluation was performed
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            wrong: List of InputExample objects corresponding to each incorrect prediction by the model
        """  # noqa: ignore flake8"

        assert len(preds) == len(labels)

        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(labels, preds)

        if multi_label:
            threshold_values = self.args.threshold if self.args.threshold else 0.5
            if isinstance(threshold_values, list):
                mismatched = labels != [
                    [self._threshold(pred, threshold_values[i]) for i, pred in enumerate(example)] for example in preds
                ]
            else:
                mismatched = labels != [
                    [self._threshold(pred, threshold_values) for pred in example] for example in preds
                ]
        else:
            mismatched = labels != preds

        if eval_examples:
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        if multi_label:
            label_ranking_score = label_ranking_average_precision_score(labels, preds)
            return {**{"LRAP": label_ranking_score}, **extra_metrics}, wrong

        elif self.args.regression[curr_task]:
            regression_metrics = {'pearson_corr': pearson_corr, 'spearman_corr': spearman_corr, 'rmse': rmse}
            reg_metrics = {}
            for metric, func in regression_metrics.items():
                reg_metrics[metric] = func(labels, preds)
            return {**reg_metrics, **extra_metrics}, wrong

        mcc = matthews_corrcoef(labels, preds)

        if self.model.num_labels[curr_task] == 2:
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            return (
                {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
                wrong,
            )
        else:
            return {**{"mcc": mcc}, **extra_metrics}, wrong

    def predict(self, to_predict, curr_task, multi_label=False):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.

        Returns:
            preds: A python list of the predictions (0 or 1) for each text.
            model_outputs: A python list of the raw model outputs for each text.
        """

        model = self.model
        args = self.args

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = np.empty((len(to_predict), self.num_labels_per_task[curr_task]))
        if multi_label:
            out_label_ids = np.empty((len(to_predict), self.num_labels_per_task[curr_task]))
        else:
            out_label_ids = np.empty((len(to_predict)))

        if not multi_label and self.args.onnx:
            model_inputs = self.tokenizer.batch_encode_plus(
                to_predict, return_tensors="pt", padding=True, truncation=True
            )

            for i, (input_ids, attention_mask) in enumerate(
                    zip(model_inputs["input_ids"], model_inputs["attention_mask"])
            ):
                input_ids = input_ids.unsqueeze(0).detach().cpu().numpy()
                attention_mask = attention_mask.unsqueeze(0).detach().cpu().numpy()
                inputs_onnx = {"input_ids": input_ids, "attention_mask": attention_mask}

                # Run the model (None = get all the outputs)
                output = self.model.run(None, inputs_onnx)

                preds[i] = output[0]
                # if preds is None:
                #     preds = output[0]
                # else:
                #     preds = np.append(preds, output[0], axis=0)

            model_outputs = preds
            preds = np.argmax(preds, axis=1)

        else:
            self._move_model_to_device()
            dummy_label = 0 if not self.args.labels_map else next(iter(self.args.labels_map.keys()))

            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            if multi_label:
                if isinstance(to_predict[0], list):
                    eval_examples = [
                        InputExample(i, text[0], text[1], [dummy_label for i in range(self.num_labels_per_task[curr_task])])
                        for i, text in enumerate(to_predict)
                    ]
                else:
                    eval_examples = [
                        InputExample(i, text, None, [dummy_label for i in range(self.num_labels_per_task[curr_task])])
                        for i, text in enumerate(to_predict)
                    ]
            else:
                if isinstance(to_predict[0], list):
                    eval_examples = [
                        InputExample(i, text[0], text[1], dummy_label) for i, text in enumerate(to_predict)
                    ]
                else:
                    eval_examples = [InputExample(i, text, None, dummy_label) for i, text in enumerate(to_predict)]
            if args.sliding_window:
                eval_dataset, window_counts = self.load_and_cache_examples(eval_examples, curr_task=curr_task, evaluate=True, no_cache=True)
                preds = np.empty((len(eval_dataset), self.num_labels_per_task[curr_task]))
                if multi_label:
                    out_label_ids = np.empty((len(eval_dataset), self.num_labels_per_task[curr_task]))
                else:
                    out_label_ids = np.empty((len(eval_dataset)))
            else:
                eval_dataset = self.load_and_cache_examples(
                    eval_examples, curr_task=curr_task, evaluate=True, multi_label=multi_label, no_cache=True
                )

            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

            if self.args.fp16:
                from torch.cuda import amp

            if self.config.output_hidden_states:
                model.eval()
                preds = None
                out_label_ids = None
                for i, batch in enumerate(eval_dataloader):
                    # batch = tuple(t.to(device) for t in batch)
                    with torch.no_grad():
                        inputs = self._get_inputs_dict(batch, curr_task)

                        if self.args.fp16:
                            with amp.autocast():
                                outputs = model(**inputs)
                                tmp_eval_loss, logits = outputs[:2]
                        else:
                            outputs = model(**inputs)
                            tmp_eval_loss, logits = outputs[:2]
                        embedding_outputs, layer_hidden_states = outputs[2][0], outputs[2][1:]

                        if multi_label:
                            logits = logits.sigmoid()

                        if self.args.n_gpu > 1:
                            tmp_eval_loss = tmp_eval_loss.mean()
                        eval_loss += tmp_eval_loss.item()

                    nb_eval_steps += 1

                    if preds is None:
                        preds = logits.detach().cpu().numpy()
                        out_label_ids = inputs["labels"].detach().cpu().numpy()
                        all_layer_hidden_states = np.array(
                            [state.detach().cpu().numpy() for state in layer_hidden_states]
                        )
                        all_embedding_outputs = embedding_outputs.detach().cpu().numpy()
                    else:
                        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                        out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                        all_layer_hidden_states = np.append(
                            all_layer_hidden_states,
                            np.array([state.detach().cpu().numpy() for state in layer_hidden_states]),
                            axis=1,
                        )
                        all_embedding_outputs = np.append(
                            all_embedding_outputs, embedding_outputs.detach().cpu().numpy(), axis=0
                        )

            else:
                n_batches = len(eval_dataloader)
                for i, batch in enumerate(eval_dataloader):

                    model.eval()
                    # batch = tuple(t.to(device) for t in batch)

                    with torch.no_grad():
                        inputs = self._get_inputs_dict(batch, curr_task)

                        if self.args.fp16:
                            with amp.autocast():
                                outputs = model(**inputs)
                                tmp_eval_loss, logits = outputs[:2]
                        else:
                            outputs = model(**inputs)
                            tmp_eval_loss, logits = outputs[:2]

                        if multi_label:
                            logits = logits.sigmoid()

                        if self.args.n_gpu > 1:
                            tmp_eval_loss = tmp_eval_loss.mean()
                        eval_loss += tmp_eval_loss.item()

                    nb_eval_steps += 1

                    start_index = self.args.eval_batch_size * i
                    end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else len(eval_dataset)
                    preds[start_index:end_index] = logits.detach().cpu().numpy()
                    out_label_ids[start_index:end_index] = inputs["labels"].detach().cpu().numpy()

                    # if preds is None:
                    #     preds = logits.detach().cpu().numpy()
                    #     out_label_ids = inputs["labels"].detach().cpu().numpy()
                    # else:
                    #     preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    #     out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps

            if args.sliding_window:
                count = 0
                window_ranges = []
                for n_windows in window_counts:
                    window_ranges.append([count, count + n_windows])
                    count += n_windows

                preds = [preds[window_range[0]: window_range[1]] for window_range in window_ranges]

                model_outputs = preds

                preds = [np.argmax(pred, axis=1) for pred in preds]
                final_preds = []
                for pred_row in preds:
                    mode_pred, counts = mode(pred_row)
                    if len(counts) > 1 and counts[0] == counts[1]:
                        final_preds.append(args.tie_value)
                    else:
                        final_preds.append(mode_pred[0])
                preds = np.array(final_preds)
            elif not multi_label and args.regression[curr_task] is True:
                preds = np.squeeze(preds)
                model_outputs = preds

            else:
                model_outputs = preds
                preds = np.argmax(preds, axis=1)

        if self.args.labels_map and not self.args.regression[curr_task]:
            inverse_labels_map = {value: key for key, value in self.args.labels_map.items()}
            preds = [inverse_labels_map[pred] for pred in preds]

        if self.config.output_hidden_states:
            return preds, model_outputs, all_embedding_outputs, all_layer_hidden_states
        else:
            return preds, model_outputs

    def convert_to_onnx(self, output_dir=None, set_onnx_arg=True):
        """Convert the model to ONNX format and save to output_dir

        Args:
            output_dir (str, optional): If specified, ONNX model will be saved to output_dir (else args.output_dir will be used). Defaults to None.
            set_onnx_arg (bool, optional): Updates the model args to set onnx=True. Defaults to True.
        """  # noqa
        if not output_dir:
            output_dir = os.path.join(self.args.output_dir, "onnx")
        os.makedirs(output_dir, exist_ok=True)

        if os.listdir(output_dir):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Output directory for onnx conversion must be empty.".format(output_dir)
            )

        onnx_model_name = os.path.join(output_dir, "onnx_model.onnx")

        with tempfile.TemporaryDirectory() as temp_dir:
            self.save_model(output_dir=temp_dir, model=self.model)

            convert(
                framework="pt",
                model=temp_dir,
                tokenizer=self.tokenizer,
                output=Path(onnx_model_name),
                pipeline_name="sentiment-analysis",
                opset=11,
            )

        self.args.onnx = True
        self.tokenizer.save_pretrained(output_dir)
        self.config.save_pretrained(output_dir)
        self.save_model_args(output_dir)

    def _threshold(self, x, threshold):
        if x >= threshold:
            return 1
        return 0

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_inputs_dict(self, batch, curr_task):
        if isinstance(batch[0], dict):
            inputs = {key: value.squeeze().to(self.device) for key, value in batch[0].items()}
            inputs["labels"] = batch[1].to(self.device)
        else:
            batch = tuple(t.to(self.device) for t in batch)

            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

            # XLM, DistilBERT and RoBERTa don't use segment_ids
            if self.args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if self.args.model_type in ["bert", "xlnet", "albert", "layoutlm"] else None
                )

        if self.args.model_type == "layoutlm":
            inputs["bbox"] = batch[4]

        inputs['curr_task'] = curr_task

        return inputs

    def _get_last_metrics(self, metric_values, curr_task):
        '''Returns metrics for the current task and adds the task to the metric name for tracking in wandb'''
        return {metric+'_task_'+str(curr_task): values[-1] for metric, values in metric_values.items()}

    def _create_training_progress_scores(self, multi_label, curr_task, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        if multi_label:
            training_progress_scores = {
                "global_step": [],
                "LRAP": [],
                "train_loss": [],
                "eval_loss": [],
                **extra_metrics,
            }
        else:
            if self.model.num_labels[curr_task] == 2:
                training_progress_scores = {
                    "global_step": [],
                    "tp": [],
                    "tn": [],
                    "fp": [],
                    "fn": [],
                    "mcc": [],
                    "train_loss": [],
                    "eval_loss": [],
                    **extra_metrics,
                }
            elif self.model.num_labels[curr_task] == 1:
                training_progress_scores = {
                    "global_step": [],
                    "train_loss": [],
                    "eval_loss": [],
                    "rmse": [],
                    "pearson_corr": [],
                    "spearman_corr": [],
                    **extra_metrics,
                }
            else:
                training_progress_scores = {
                    "global_step": [],
                    "mcc": [],
                    "train_loss": [],
                    "eval_loss": [],
                    **extra_metrics,
                }

        return training_progress_scores

    def save_model(self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            self.save_model_args(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = MultiTransQuestArgs()
        args.load(input_dir)
        return args

    def _get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]
