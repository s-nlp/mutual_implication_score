# Benchmarking

The file `evaluation.py` contains code (very simple) for scoring a metric against all datasets.
The files `score_mis.py` and `score_new_metric.py` produce examples of how a metric can be scored.
They print the scores to the console output, however, one could save the scores to a ranking table of metrics.
The file `rank_metrics.py` shows an example of this: it ranks 5 metrics on all datasets,
and outputs the joint ranking, and ranking by mean scores on the paraphrase and text style transfer subtasks.

for example, the command `python rank_metrics.py --sample=100` would print to the console 
the following ranking:
```
ALL DATASETS
                CAE   PG_YELP  STRAP_COHA  STRAP_formality  STRAP_shakespeare    advers      etpc       msr      paws       pit      sick    tox600  yamsh_bible  yamsh_gyafc  yamsh_para  yamsh_yelp      mean
MIS        0.415549  0.214878    0.466443         0.703353           0.718589  0.559095  0.384958  0.244989  0.617526  0.607611  0.852189  0.609960     0.594375     0.682743    0.612201    0.471645  0.547256
SIMCSE-SL  0.468069  0.368665    0.428248         0.654361           0.736303  0.392107  0.392581  0.375162  0.397806  0.584583  0.848232  0.443550     0.618693     0.688844    0.662353    0.423650  0.530200
BLEURT     0.529362  0.380711    0.363089         0.623737           0.704316  0.358554  0.384958  0.448292  0.357221  0.458268  0.732774  0.621776     0.594726     0.697860    0.604586    0.394109  0.515896
chrF       0.516965  0.310940    0.171998         0.306393           0.361257  0.139287  0.423073  0.414652  0.336228  0.392393  0.543309  0.244602     0.655907     0.610148    0.491984    0.417644  0.396049
BLEU       0.414922  0.336272    0.017742         0.094107           0.228750  0.178425  0.372832  0.369311  0.352673  0.321114  0.436849  0.102910     0.586571     0.605118   -0.024668    0.356925  0.296866

PARAPHRASE DETECTION DATASETS
             advers      etpc       msr      paws       pit      sick  yamsh_para      mean
MIS        0.559095  0.384958  0.244989  0.617526  0.607611  0.852189    0.612201  0.554081
SIMCSE-SL  0.392107  0.392581  0.375162  0.397806  0.584583  0.848232    0.662353  0.521832
BLEURT     0.358554  0.384958  0.448292  0.357221  0.458268  0.732774    0.604586  0.477808
chrF       0.139287  0.423073  0.414652  0.336228  0.392393  0.543309    0.491984  0.391561
BLEU       0.178425  0.372832  0.369311  0.352673  0.321114  0.436849   -0.024668  0.286648

TEXT STYLE TRANSFER DATASETS
                CAE   PG_YELP  STRAP_COHA  STRAP_formality  STRAP_shakespeare    tox600  yamsh_bible  yamsh_gyafc  yamsh_yelp      mean
BLEURT     0.529362  0.380711    0.363089         0.623737           0.704316  0.621776     0.594726     0.697860    0.394109  0.545521
MIS        0.415549  0.214878    0.466443         0.703353           0.718589  0.609960     0.594375     0.682743    0.471645  0.541948
SIMCSE-SL  0.468069  0.368665    0.428248         0.654361           0.736303  0.443550     0.618693     0.688844    0.423650  0.536709
chrF       0.516965  0.310940    0.171998         0.306393           0.361257  0.244602     0.655907     0.610148    0.417644  0.399539
BLEU       0.414922  0.336272    0.017742         0.094107           0.228750  0.102910     0.586571     0.605118    0.356925  0.304813
```

# About the datasets

Most of the data (16 out of 19 datasets) used in the paper is already fully available in a unified format
in the folder `datasets`.  The notebook `data_preparation.ipynb` contains the code and scripts
that reproduce creation of the files in the `datasets` folder.

Some datasets used in the paper ([PAWS-QQP](https://github.com/google-research-datasets/paws), 
[Twitter-URL](https://github.com/lanwuwei/Twitter-URL-Corpus), 
[xformal-FoST](https://github.com/Elbria/xformal-FoST-meta))
are not included, because one has to perform extra actions to obtain them: PAWS-QQP should be generated
from scratch, and Twitter-URL and xformal-FoST should be requested from their authors.

Most of the datasets were taken from their open-access sources: 
[APT](https://github.com/Advancing-Machine-Human-Reasoning-Lab/apt),
[ETPC](https://github.com/venelink/ETPC),
[MSR](https://github.com/brmson/dataset-sts/tree/master/data/para/msr),
[PAWS](https://github.com/google-research-datasets/paws),
[PIT](https://github.com/cocoxu/SemEval-PIT2015),
[SICK](https://github.com/brmson/dataset-sts/tree/master/data/sts/sick2014),
[STRAP](https://github.com/martiansideofthemoon/style-transfer-paraphrase),
[Tox600](https://github.com/skoltech-nlp/detox),
and [datasets](https://github.com/VAShibaev/semantic_similarity_metrics) 
from [Yamshchikov et al, 2020](https://arxiv.org/abs/2004.05001).

The folder `source_data` includes some of the datasets that were not published
before, but their authors allowed us to use them. Specifically,  
`textual-transfer-pairwise-comparison` contains the data from the paper 
[Unsupervised Evaluation Metrics and Learning Criteria for Non-Parallel Textual Transfer](https://aclanthology.org/D19-5614/) 
by Pang and Gimpel (2019), and `f1659626_annon.csv` is from 
[Civil Rephrases Of Toxic Texts With Self-Supervised Transformers](https://aclanthology.org/2021.eacl-main.124/)
by Laugier et al. (2021). We use these datasets under the names PG_YELP and CAE, respectively.

In two datasets (ETPC and MSR), we found and fixed faults after publishing
the paper, so the scores on them could be different from the ones in the paper.


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
