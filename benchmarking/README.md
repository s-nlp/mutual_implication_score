# Benchmarking

The file `evaluation.py` contains code (very simple) for scoring a metric against all datasets.
The files `score_mis.py` and `score_new_metric.py` produce examples of how a metric can be scored.
They print the scores to the console output, however, one could save the scores to a ranking table of metrics.
The file `rank_metrics.py` shows an example of this: it ranks 5 metrics on all datasets,
and outputs the joint ranking, and ranking by mean scores on the paraphrase and text style transfer subtasks.

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
