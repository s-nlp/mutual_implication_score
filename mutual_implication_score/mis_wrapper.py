from typing import List

import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from .mis_architecture import TwoFoldRoberta


class PairsDatasetInference(Dataset):
    def __init__(self, texts_first, texts_second):
        self.texts_first = texts_first
        self.texts_second = texts_second

    def __len__(self):
        return len(self.texts_first)

    def __getitem__(self, idx):
        return self.texts_first[idx], self.texts_second[idx]


class MIS:
    def __init__(
        self,
        model_name='SkolkovoInstitute/Mutual_Implication_Score',
        device=None
    ):
        """
            model_name: name or path to the pretrained model (by default, 'SkolkovoInstitute/Mutual_Implication_Score')
            device: name of the pytorch device (by default, 'cuda' or 'cpu', if CUDA is not available)
        """
        self.model_name = model_name
        self.device = torch.device(device or 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TwoFoldRoberta.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def compute(self, source_texts, paraphrases, verbose=True, batch_size=16) -> List[float]:
        dataset_direct = PairsDatasetInference(source_texts, paraphrases)
        dataloader_direct = DataLoader(dataset_direct, batch_size=batch_size)

        dataset_reverse = PairsDatasetInference(paraphrases, source_texts)
        dataloader_reverse = DataLoader(dataset_reverse, batch_size=batch_size)

        iterator = zip(dataloader_direct, dataloader_reverse)
        if verbose:
            iterator = tqdm(iterator, total=len(dataloader_reverse))

        preds = []

        for b1, b2 in iterator:
            with torch.no_grad():
                tokenized1 = self.tokenizer(*b1, padding=True, truncation='longest_first',
                                            return_tensors="pt").to(self.device)
                tokenized2 = self.tokenizer(*b2, padding=True, truncation='longest_first',
                                            return_tensors="pt").to(self.device)
                merged_prob = self.model(tokenized1, tokenized2)
                merged_prob = torch.sigmoid(merged_prob)

            merged_prob = merged_prob.cpu().numpy()
            preds.extend(merged_prob)

        preds_float = [float(e) for e in preds]
        return preds_float
