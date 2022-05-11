from evaluation import score_metric, vectorize_metric


def text2bow(text):
    return set(text.lower().split())


def bow_jaccard_metric(text1, text2):
    b1 = text2bow(text1)
    b2 = text2bow(text2)
    return len(b1.intersection(b2)) / max(1, len(b1.union(b2)))


if __name__ == '__main__':
    print(score_metric(vectorize_metric(bow_jaccard_metric)))
