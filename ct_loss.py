import copy
from typing import Dict, Iterable

import torch
from torch import Tensor, nn


class ContrastiveTensionLoss(nn.Module):
    def __init__(self, model):
        super(ContrastiveTensionLoss, self).__init__()
        self.model1 = (
            model  # This will be the final model used during the inference time.
        )
        self.model2 = copy.deepcopy(model)
        self.criterion = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        sentence_features1, sentence_features2 = tuple(sentence_features)
        reps_1 = self.model1(sentence_features1)["sentence_embedding"]  # (bsz, hdim)
        reps_2 = self.model2(sentence_features2)["sentence_embedding"]

        sim_scores = (
            torch.matmul(reps_1[:, None], reps_2[:, :, None]).squeeze(-1).squeeze(-1)
        )  # (bsz,) dot product, i.e. S1S2^T

        loss = self.criterion(sim_scores, labels.type_as(sim_scores))
        return loss
