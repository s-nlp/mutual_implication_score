import pytest


def test_basic():
    from mutual_implication_score import MIS
    mis = MIS(device='cpu')
    source_texts = ['I want to leave this room',
                    'Hello world, my name is Nick']
    paraphrases = ['I want to go out of this room',
                   'Hello world, my surname is Petrov']
    scores = mis.compute(source_texts, paraphrases)
    assert scores == pytest.approx([0.9748, 0.0545], abs=1e-4)
