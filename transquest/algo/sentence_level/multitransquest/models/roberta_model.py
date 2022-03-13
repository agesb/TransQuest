import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import (
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    RobertaClassificationHead,
    RobertaConfig,
    RobertaModel,
)

from CODE.transquest.algo.sentence_level.multitransquest.grad_reversal import WeightGradientsFunc, WeightGradients, test_weight_gradients

class RobertaForMultitaskSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_tasks, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """  # noqa: ignore flake8"
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "roberta"

    def __init__(self, config, weight=None):
        super(RobertaForMultitaskSequenceClassification, self).__init__(config)
        # TODO This is now a list, not a single value – CAN I ACTUASLLY GET T>HAT FROM CONFIG? otherwise i might have to pass it down
        self.num_tasks = self.config.num_tasks
        self.num_labels = self.config.list_labels
        # self.grad_weights = self.config.grad_weights

        self.roberta = RobertaModel(config)
        # The classifier has to support multiple classification / regression heads
        # TODO non hard coded
        self.classifier = RobertaClassificationMultitaskHead(config)
        self.weight = weight

    def forward(
            self,
            # The forward method needs to know which head is associated with the training batch
            curr_task,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            encode_decode='both',
            encoded_features=None,
    ):
        if encode_decode == 'encode':
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
            )

            sequence_output = outputs[0]

            return sequence_output

        elif encode_decode == 'decode':

            logits = self.classifier(encoded_features, curr_task)

            # TODO: Question: can I ignore the pooler stuff?
            outputs = (logits,)
            if labels is not None:
                if self.num_labels[curr_task] == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    if self.weight is not None:
                        weight = self.weight.to(labels.device)
                    else:
                        weight = None
                    loss_fct = CrossEntropyLoss(weight=weight)
                    loss = loss_fct(logits.view(-1, self.num_labels[curr_task]), labels.view(-1))
                outputs = (loss,) + outputs

            return outputs  # (loss), logits, (hidden_states), (attentions)

        else:
            if not encode_decode == 'both':
                print("Attention: assuming that a complete forward pass is to be made.")
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
            )
            sequence_output = outputs[0]
            # The classifier requires information on the current head
            logits = self.classifier(sequence_output, curr_task)

            outputs = (logits,) + outputs[2:]
            if labels is not None:
                if self.num_labels[curr_task] == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    if self.weight is not None:
                        weight = self.weight.to(labels.device)
                    else:
                        weight = None
                    loss_fct = CrossEntropyLoss(weight=weight)
                    loss = loss_fct(logits.view(-1, self.num_labels[curr_task]), labels.view(-1))
                outputs = (loss,) + outputs

            return outputs  # (loss), logits, (hidden_states), (attentions)

class RobertaClassificationMultitaskHead(nn.Module):
    """Head for sentence-level classification and regression multi-tasking."""

    def __init__(self, config):
        super(RobertaClassificationMultitaskHead, self).__init__()

        # Get relevant config params
        self.num_tasks = config.num_tasks
        self.num_labels = config.list_labels
        self.grad_weights = config.grad_weights
        print("Grad_weights:", self.grad_weights)
        print("num_labels:", self.num_labels)
        self.num_shared_decoder_layers = config.num_shared_decoder_layers
        self.num_separate_decoder_layers = config.num_separate_decoder_layers

        # Initiate the shared decoder layers with as many layers as specified in the config
        self.shared_layers_decoder = torch.nn.Sequential()
        if self.num_shared_decoder_layers > 0:
            for layers in range(self.num_shared_decoder_layers):
                self.shared_layers_decoder.add_module("dense_" + str(layers),
                                                      nn.Linear(config.hidden_size, config.hidden_size))
                self.shared_layers_decoder.add_module("tanh_" + str(layers), nn.Tanh())
                self.shared_layers_decoder.add_module("dropout_" + str(layers), nn.Dropout(config.hidden_dropout_prob))

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initiate the separate gradient weighting & linear output projection layers
        self.weight_grads_per_head = nn.ModuleList([WeightGradients(weight) for weight in self.grad_weights])
        if self.num_separate_decoder_layers > 0:
            self.linear_per_head_1 = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for i in range(self.num_tasks)])
        if self.num_separate_decoder_layers > 1:
            self.linear_per_head_2 = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for i in range(self.num_tasks)])
        self.out_proj_per_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, self.num_labels[i]) for i in range(self.num_tasks)])

    def forward(self, features, curr_task, **kwargs):
        # Shared layers  that take the <s> token as input (equiv. to [CLS])
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        # Pass the encoded features through the shared decoder layers
        if self.num_shared_decoder_layers > 0:
            x = self.shared_layers_decoder(x)

        # Separate layers for each head - as of now, the architecture is identical for all heads and linear.
        x = self.weight_grads_per_head[curr_task](x)

        # One nonlinear layer:
        if self.num_separate_decoder_layers > 0:
            x = self.linear_per_head_1[curr_task](x)
            x = torch.tanh(x)
            x = self.dropout(x)

        # Two nonlinear layers:
        if self.num_separate_decoder_layers > 1:
            x = self.linear_per_head_2[curr_task](x)
            x = torch.tanh(x)
            x = self.dropout(x)
        if self.num_separate_decoder_layers > 3:
            print("Warning: Please note that we only experimented with up tp two separate decoder layers. \
                  If you would like to try three or more, please modify roberta_model.py.\
                  The code is now executed with 2 layers.")


        # Output projection
        x = self.out_proj_per_head[curr_task](x)

        return x
