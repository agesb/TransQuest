from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import (
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    RobertaClassificationHead,
    RobertaConfig,
    RobertaModel,
)


class RobertaForSequenceClassification(BertPreTrainedModel):
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
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
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
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.weight = weight

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss(reduction='none')
                loss_per_sample = loss_fct(logits.view(-1), labels.view(-1))
                loss = loss_per_sample.mean()
            else:
                if self.weight is not None:
                    weight = self.weight.to(labels.device)
                else:
                    weight = None
                loss_fct = CrossEntropyLoss(weight=weight, reduction='none')
                loss_per_sample = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss = loss_per_sample.mean()

            outputs = (loss,) + outputs + (loss_per_sample,)

        return outputs  # (loss), logits, (hidden_states), (attentions)
