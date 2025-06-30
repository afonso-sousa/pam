import json
import os
from typing import Dict

import torch
from sentence_transformers.models import Pooling
from torch import Tensor


class SimplePooling(Pooling):
    """Performs pooling (max or mean) on the token embeddings without using attention.

    This version assumes no padding tokens or treats all tokens equally.
    """

    def __init__(
        self,
        word_embedding_dimension: int,
        pooling_mode: str = None,
        pooling_mode_cls_token: bool = False,
        pooling_mode_max_tokens: bool = False,
        pooling_mode_mean_tokens: bool = True,
        pooling_mode_mean_sqrt_len_tokens: bool = False,
        pooling_mode_weightedmean_tokens: bool = False,
        pooling_mode_lasttoken: bool = False,
        include_prompt: bool = True,
    ):
        super(SimplePooling, self).__init__(
            word_embedding_dimension,
            pooling_mode,
            pooling_mode_cls_token,
            pooling_mode_max_tokens,
            pooling_mode_mean_tokens,
            pooling_mode_mean_sqrt_len_tokens,
            pooling_mode_weightedmean_tokens,
            pooling_mode_lasttoken,
            include_prompt,
        )

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features["token_embeddings"]

        # Pooling strategy without attention masks
        output_vectors = []
        if self.pooling_mode_cls_token:
            cls_token = features.get(
                "cls_token_embeddings", token_embeddings[:, 0]
            )  # Take the first token as CLS token by default
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            max_over_time = torch.max(token_embeddings, 1)[
                0
            ]  # Take the max over the sequence dimension
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            sum_embeddings = torch.sum(token_embeddings, 1)
            seq_len = token_embeddings.size(
                1
            )  # Length of the sequence (number of tokens)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / seq_len)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(
                    sum_embeddings
                    / torch.sqrt(
                        torch.tensor(
                            seq_len, dtype=torch.float32, device=token_embeddings.device
                        )
                    )
                )
        if self.pooling_mode_weightedmean_tokens:
            weights = (
                torch.arange(start=1, end=token_embeddings.shape[1] + 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
                .to(token_embeddings.device)
            )
            weighted_sum_embeddings = torch.sum(token_embeddings * weights, 1)
            sum_weights = torch.sum(weights, 1)
            output_vectors.append(weighted_sum_embeddings / sum_weights)
        if self.pooling_mode_lasttoken:
            last_token = token_embeddings[
                :, -1
            ]  # Select the last token in the sequence
            output_vectors.append(last_token)

        output_vector = torch.cat(output_vectors, 1)
        features.update({"sentence_embedding": output_vector})
        return features

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        return SimplePooling(**config)
