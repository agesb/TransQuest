[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
# Bias Mitigation in Machine Translation Quality Estimation

Machine Translation Quality Estimation (QE) aims to build predictive models to assess the quality of machine-generated translations in the absence of reference translations. While state-of-the-art QE models have been shown to achieve good results, they over-rely on features that do not have a causal impact on the quality of a translation. In particular, there appears to be a partial input bias, i.e., a tendency to assign high-quality scores to translations that are fluent and grammatically correct, even though they do not preserve the meaning of the source. We analyse the partial input bias in further detail and evaluate four approaches to use auxiliary tasks for bias mitigation. Two approaches use additional data to inform and support the main task, while the other two are adversarial, actively discouraging the model from learning the bias. We compare the methods with respect to their ability to reduce the partial input bias while maintaining the overall performance. We find that training a multitask architecture with an auxiliary binary classification task that utilises additional augmented data achieves the desired effects and generalises well to different languages and quality metrics.

This repository builds upon the state-of-the-art QE model MonoTransQuest and includes all modifications and added functionality that were made as part of this research project. In particular, it includes MultiTransQuest, an alternative architecture that can be trained with multiple auxiliary tasks.



## Navigating the Repository

The repository features two architectures for sentence-level translation quality estimation:

- **MonoTransQuest**, the original architecture proposed by Ranasinghe *et al*, including modifications to enable training with focal loss. All relevant files for the MonoTransQuest architecture are located in the folder */transquest/algo/sentence_level/monotransquest*.

- **MultiTransQuest**, a modified version of MonoTransQuest to allow training with auxiliary tasks in a multitask setup. All relevant files for the MultiTransQuest architecture are located in the folder *transquest/algo/sentence_level/multitransquest*.

In addition to the architectures, the folder *transquest/training* includes additional **utility functions for training**, including the functions written to execute the different experiments. Most importantly, it also holds the main config files  *monotransquest_config.py* and *multitransquest_config.py* used to control the hyperparameters, including path variables that need to be adjusted to the local project structure.

Trained models are automatically saved to *temp*.

The folder *examples* holds four .py files that can be used to replicate the experiments discussed in the paper. We worked with publicly available data, only:
- [MLQE-PE](https://github.com/sheffieldnlp/mlqe-pe)
- [WikiMatrix](https://github.com/facebookresearch/LASER/tree/main/tasks/WikiMatrix)

The folder *data* holds the EN-DE test set from the MLQE-PE dataset, annotated to highlight fluent but inadequate translations (the labelling was done as part of the research project).

**Important:** Please note that the training process involves loading and training XLM-R base. We recommend training the models on a GPU. Saving the models consumes up to 2 GB of storage.


## Citations

*Coming soon*
```
