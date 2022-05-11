import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

from benchmarking.evaluation import score_metric
from mutual_implication_score import MIS
from tqdm.auto import tqdm, trange
from sacrebleu.metrics import BLEU, CHRF


def make_mis():
    scorer = MIS()
    return scorer.compute


def make_bleurt():
    model_path = 'Elron/bleurt-large-128'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    if torch.cuda.is_available():
        model.cuda()

    def apply(texts1, texts2, batch_size=16):
        results = []
        for i in trange(0, len(texts1), batch_size):
            inputs = tokenizer(
                texts1[i:i+batch_size], texts2[i:i+batch_size],
                padding=True, truncation=True, return_tensors="pt"
            ).to(model.device)
            with torch.inference_mode():
                result = model(**inputs)[0].cpu().numpy()
            results.extend(result.ravel())
        return results

    return apply


def cos(vecs1, vecs2):
    return np.sum(vecs1 * vecs2, axis=1) / np.sqrt(np.sum(vecs1**2, axis=1) * np.sum(vecs1**2, axis=1))


def make_simcse():
    model_path = 'princeton-nlp/sup-simcse-roberta-large'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    if torch.cuda.is_available():
        model.cuda()

    def apply(texts1, texts2, batch_size=16):
        results = []
        for i in trange(0, len(texts1), batch_size):
            vecs = []
            for source in texts1, texts2:
                inputs = tokenizer(
                    source[i:i+batch_size],
                    padding=True, truncation=True, return_tensors="pt"
                ).to(model.device)
                with torch.inference_mode():
                    result = model(**inputs).pooler_output.cpu().numpy()
                vecs.append(result)
            results.extend(cos(*vecs))
        return results

    return apply


def make_bleu():
    bleu = BLEU()

    def apply(texts1, texts2):
        return [bleu.corpus_score([t1], [[t2]]).score for t1, t2 in zip(texts1, texts2)]

    return apply


def make_chrf():
    scorer = CHRF()

    def apply(texts1, texts2):
        return [scorer.corpus_score([t1], [[t2]]).score for t1, t2 in zip(texts1, texts2)]

    return apply


def is_tst(dataset_name):
    return dataset_name.split('.')[0] in {
        'CAE', 'PG_YELP', 'STRAP_COHA', 'STRAP_formality', 'STRAP_shakespeare', 'tox600',
        'yamsh_bible', 'yamsh_gyafc', 'yamsh_yelp'
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='datasets', help='path to the dataset(s) to score the metrics')
    parser.add_argument(
        '--sample', type=int, default=None, help='sample size to score on (for speedup), by default use all data'
    )
    args = parser.parse_args()

    results = {}
    for k, v in {
        'MIS': make_mis,
        'BLEURT': make_bleurt,
        'SIMCSE-SL': make_simcse,
        'BLEU': make_bleu,
        'chrF': make_chrf,
    }.items():
        print(k)
        results[k] = score_metric(v(), data=args.data, sample=args.sample, verbose=True)
    df = pd.DataFrame(results).T
    tst_cols = [c for c in df.columns if is_tst(c)]
    df_tst = df[tst_cols].copy()
    pd_cols = [c for c in df.columns if not is_tst(c)]
    df_pd = df[pd_cols].copy()
    df['mean'] = df.mean(axis=1)
    df_pd['mean'] = df_pd.mean(axis=1)
    df_tst['mean'] = df_tst.mean(axis=1)

    print('ALL DATASETS')
    print(df.sort_values('mean', ascending=False))
    print('PARAPHRASE DETECTION DATASETS')
    print(df_pd.sort_values('mean', ascending=False))
    print('TEXT STYLE TRANSFER DATASETS')
    print(df_tst.sort_values('mean', ascending=False))
