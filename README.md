This repository presents the results of [A large-scale computational study of content preservation measures for text style transfer and paraphrase generation](https://aclanthology.org/2022.acl-srw.23/). It consists of two parts: 
- code for usage of Mutual Implication Score (text similarity measure which demonstrates SOTA-performance of paraphrases generation task and performs on par with SOTA measure on text style transfer task)
- code and dataset for reproducing  a large-scale comparison of different text similarity measures


# Mutual Implication Score

## Model overview

Mutual Implication Score is a symmetric measure of text semantic similarity
based on a RoBERTA model pretrained for natural language inference
and fine-tuned on paraphrases dataset. 

It is **particularly useful for paraphrases detection**, but can also be applied to other semantic similarity tasks, such as text style transfer.

## How to use
The following snippet illustrates code usage:
```python
!pip install mutual-implication-score

from mutual_implication_score import MIS
mis = MIS(device='cpu')
source_texts = ['I want to leave this room',
                'Hello world, my name is Nick']
paraphrases = ['I want to go out of this room',
               'Hello world, my surname is Petrov']
scores = mis.compute(source_texts, paraphrases)
print(scores)
# expected output: [0.9748, 0.0545]
```

## Model details

We slightly modify [RoBERTa-Large NLI](https://huggingface.co/ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli) model architecture (see the scheme below) and fine-tune it with [QQP](https://www.kaggle.com/c/quora-question-pairs) paraphrases dataset.

![alt text](https://github.com/skoltech-nlp/mutual_implication_score/blob/main/MIS.jpg)


## Performance on Text Style Transfer and Paraphrase Detection tasks

This measure was developed in terms of large scale comparison of different measures on text style transfer and paraphrases datasets.

<img src="https://github.com/skoltech-nlp/mutual_implication_score/blob/main/corr_main.jpg" alt="drawing" width="1000"/>

The scheme above shows the correlations of measures of different classes with human judgments on paraphrase and text style transfer datasets. The text above each dataset indicates the best-performing measure. The rightmost columns show the mean performance of measures across the datasets.

MIS outperforms all measures on paraphrases detection task and performs on par with top measures on text style transfer task. 

To learn more refer to our article: [A large-scale computational study of content preservation measures for text style transfer and paraphrase generation](https://aclanthology.org/2022.acl-srw.23/)


# Measures comparison

One of the main impacts of our work is a large scale comparison of 57 measure on text style transfer and paraphrases detection tasks. You can reproduce the subset of our computations using the code and and most of the datasets in `benchmarking` folder. 
More details are in [README](https://github.com/skoltech-nlp/mutual_implication_score/blob/main/benchmarking/README.md).

# Citations

If you find this repository helpful, feel free to cite our publication:

```
@inproceedings{babakov-etal-2022-large,
    title = "A large-scale computational study of content preservation measures for text style transfer and paraphrase generation",
    author = "Babakov, Nikolay  and
      Dale, David  and
      Logacheva, Varvara  and
      Panchenko, Alexander",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-srw.23",
    pages = "300--321",
    abstract = "Text style transfer and paraphrasing of texts are actively growing areas of NLP, dozens of methods for solving these tasks have been recently introduced. In both tasks, the system is supposed to generate a text which should be semantically similar to the input text. Therefore, these tasks are dependent on methods of measuring textual semantic similarity. However, it is still unclear which measures are the best to automatically evaluate content preservation between original and generated text. According to our observations, many researchers still use BLEU-like measures, while there exist more advanced measures including neural-based that significantly outperform classic approaches. The current problem is the lack of a thorough evaluation of the available measures. We close this gap by conducting a large-scale computational study by comparing 57 measures based on different principles on 19 annotated datasets. We show that measures based on cross-encoder models outperform alternative approaches in almost all cases.We also introduce the Mutual Implication Score (MIS), a measure that uses the idea of paraphrasing as a bidirectional entailment and outperforms all other measures on the paraphrase detection task and performs on par with the best measures in the text style transfer task.",
}
```
