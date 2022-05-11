from evaluation import score_metric
from mutual_implication_score import MIS  # we assume the package is installed

if __name__ == '__main__':
    scorer = MIS()
    print(score_metric(scorer.compute))
