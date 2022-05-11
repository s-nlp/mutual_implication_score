# mutual_implication_score

Mutual implication score: a symmetric measure of text semantic similarity
based on a RoBERTA model pretrained for natural language inference
and fine-tuned for paraphrase detection.

The following snippet illustrates code usage:
```python
from mutual_implication_score import MIS
mis = MIS(device='cpu')
source_texts = ['I want to leave this room', 'Hello world, my name is Nick']
paraphrases = ['I want to go out of this room', 'Hello world, my surname is Petrov']
scores = mis.compute(source_texts, paraphrases)
print(scores)
# expected output: [0.9748, 0.0545]
```

The first two texts are semantically equivalent, their MIS is close to 1. 
The two other texts have different meanings, and their score is low.

By default, the model 
https://huggingface.co/SkolkovoInstitute/Mutual_Implication_Score
is used, but you can provide any other compatible model.

# benchmarking

The `benchmarking` folder contains the code and most of the datasets used in the study of content similarity measures.
More details are in its README file.
