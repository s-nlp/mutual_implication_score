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
    save_directory=None,
    metric_name=None,
    sample=None,
):
    """
    Arguments:
        metric_applier: function with signature (texts1: List[str], texts2: List[str]) -> scores: List[float]
        data: optional, path to the csv file or folder with such files, with texts to score.
              The file should have columns 'text1', 'text2' and 'human'. Default: 'datasets'.
        verbose: optional, whether to print filenames before calculation
        save_directory: optional, the directory to save the files with scores
        metric_name: optional, the column name for the metric in the saved files with scores
        sample: optional,
    """
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
        if sample and (df.shape[0] > sample):
            df = df.sample(sample, random_state=1)
        values = metric_applier(df['text1'].tolist(), df['text2'].tolist())
        score = scipy.stats.spearmanr(df['human'], values).correlation
        result[os.path.splitext(os.path.basename(fn))[0]] = score
        if save_directory and metric_name:
            new_fn = os.path.join(save_directory, os.path.basename(fn))
            if os.path.exists(new_fn):
                df = pd.read_csv(new_fn)
            df[metric_name] = values
            df.to_csv(new_fn, index=None)
    return result
