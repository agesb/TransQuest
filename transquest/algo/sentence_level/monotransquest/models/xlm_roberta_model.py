# COLAB
'''
PROJECT_PATH='drive/MyDrive/Adversarial_MTQE/'
import sys
sys.path.append(PROJECT_PATH+'CODE/transquest/algo/sentence_level/monotransquest/models')
from xlm_model import XLMForSequenceClassification
from roberta_model import RobertaForSequenceClassification
'''

# LAB MACHINE
#PROJECT_PATH='/vol/bitbucket/hsb20/'

from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST

from CODE.transquest.algo.sentence_level.monotransquest.models.roberta_model import RobertaForSequenceClassification

class XLMRobertaForSequenceClassification(RobertaForSequenceClassification):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
