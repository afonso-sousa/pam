import csv
import logging
import os
from typing import List

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
)
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

from input_example import InputExample

logger = logging.getLogger(__name__)


class MyEmbeddingSimilarityEvaluator(EmbeddingSimilarityEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(
        self,
        sentences1: List[str],
        sentences2: List[str],
        scores: List[float],
        batch_size: int = 16,
        main_similarity: SimilarityFunction = None,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
        ref1_graphs_index=None,
        ref1_graphs_type=None,
        ref2_graphs_index=None,
        ref2_graphs_type=None,
        ref1_pos_ids=None,
        ref2_pos_ids=None,
    ):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param sentences1:  List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param scores: Similarity score between sentences1[i] and sentences2[i]
        :param write_csv: Write results to a CSV file
        """
        super().__init__(
            sentences1,
            sentences2,
            scores,
            batch_size,
            main_similarity,
            name,
            show_progress_bar,
            write_csv,
        )

        self.ref1_graphs_index = ref1_graphs_index
        self.ref1_graphs_type = ref1_graphs_type
        self.ref2_graphs_index = ref2_graphs_index
        self.ref2_graphs_type = ref2_graphs_type
        self.ref1_pos_ids = ref1_pos_ids
        self.ref2_pos_ids = ref2_pos_ids

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []
        ref1_graphs_index = []
        ref1_graphs_type = []
        ref2_graphs_index = []
        ref2_graphs_type = []
        ref1_pos_ids = []
        ref2_pos_ids = []
        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
            if example.edge_index:
                ref1_graphs_index.append(example.edge_index[0])
                ref1_graphs_type.append(example.edge_type[0])
                ref2_graphs_index.append(example.edge_index[1])
                ref2_graphs_type.append(example.edge_type[1])
                ref1_pos_ids.append(example.pos_ids[0])
                ref2_pos_ids.append(example.pos_ids[1])

        return cls(
            sentences1,
            sentences2,
            scores,
            ref1_graphs_index=ref1_graphs_index,
            ref1_graphs_type=ref1_graphs_type,
            ref2_graphs_index=ref2_graphs_index,
            ref2_graphs_type=ref2_graphs_type,
            ref1_pos_ids=ref1_pos_ids,
            ref2_pos_ids=ref2_pos_ids,
            **kwargs
        )

    def __call__(
        self, model, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info(
            "EmbeddingSimilarityEvaluator: Evaluating the model on "
            + self.name
            + " dataset"
            + out_txt
        )

        embeddings1 = model.encode(
            self.sentences1,
            graph_index=self.ref1_graphs_index,
            graph_type=self.ref1_graphs_type,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            pos_ids=self.ref1_pos_ids,
        )
        embeddings2 = model.encode(
            self.sentences2,
            graph_index=self.ref2_graphs_index,
            graph_type=self.ref2_graphs_type,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            pos_ids=self.ref2_pos_ids,
        )

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

        labels = self.scores
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [
            np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)
        ]

        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)

        logger.info(
            "Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
                eval_pearson_cosine, eval_spearman_cosine
            )
        )
        logger.info(
            "Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
                eval_pearson_manhattan, eval_spearman_manhattan
            )
        )
        logger.info(
            "Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
                eval_pearson_euclidean, eval_spearman_euclidean
            )
        )
        logger.info(
            "Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
                eval_pearson_dot, eval_spearman_dot
            )
        )

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(
                csv_path,
                newline="",
                mode="a" if output_file_exists else "w",
                encoding="utf-8",
            ) as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow(
                    [
                        epoch,
                        steps,
                        eval_pearson_cosine,
                        eval_spearman_cosine,
                        eval_pearson_euclidean,
                        eval_spearman_euclidean,
                        eval_pearson_manhattan,
                        eval_spearman_manhattan,
                        eval_pearson_dot,
                        eval_spearman_dot,
                    ]
                )
        if self.main_similarity == SimilarityFunction.COSINE:
            return eval_pearson_cosine, cosine_scores
        elif self.main_similarity == SimilarityFunction.EUCLIDEAN:
            return eval_spearman_euclidean
        elif self.main_similarity == SimilarityFunction.MANHATTAN:
            return eval_spearman_manhattan
        elif self.main_similarity == SimilarityFunction.DOT_PRODUCT:
            return eval_spearman_dot
        elif self.main_similarity is None:
            return max(
                eval_spearman_cosine,
                eval_spearman_manhattan,
                eval_spearman_euclidean,
                eval_spearman_dot,
            )
        else:
            raise ValueError("Unknown main_similarity value")
