import os
import pandas as pd
import scipy.stats


def vectorize_metric(single_pair_metric):
    def metric_applier(texts1, texts2):
        return [single_pair_metric(t1, t2) for t1, t2 in zip(texts1, texts2)]
    return metric_applier


def score_metric(
    metric_applier,
    data='datasets',
    verbose=True,
):
    if os.path.isfile(data):
        files = [data]
    else:
        files = [os.path.join(data, fn) for fn in sorted(os.listdir(data))]
        files = [fn for fn in files if os.path.isfile(fn)]
    result = {}
    for fn in files:
        if verbose:
            print('computing scores on', os.path.basename(fn), '...')
            df = pd.read_csv(fn)
            values = metric_applier(df.text1.tolist(), df.text2.tolist())
            score = scipy.stats.spearmanr(df['human'], values).correlation
            result[os.path.basename(fn)] = score
    return result
