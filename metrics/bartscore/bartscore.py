"""
A HuggingFace wrapper for the BARTScore metric (https://github.com/neulab/BARTScore)
"""

import logging

import datasets
import evaluate
import torch

from metrics.bartscore.bart_score import BARTScorer

logger = logging.getLogger(__name__)

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:metric,
title = {A great new metric},
authors={huggingface, Inc.},
year={2020}
}
"""

# TODO: Add description of the metric here
_DESCRIPTION = """\
This new metric is designed to solve this great NLP task and is crafted with a lot of care.
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.
    >>> my_new_metric = datasets.load_metric("my_new_metric")
    >>> results = my_new_metric.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BartScore(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
            # Homepage of the metric for documentation
            homepage="http://metric.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=["http://path.to.reference.url/new_metric"],
        )

    def _download_and_prepare(self, dl_manager):
        self.bart_score = BARTScorer(checkpoint="facebook/bart-large-cnn")

    def _compute(self, predictions, references, batch_size=64):
        # ref to hypo scores are the precision
        ref_hypo_scores = self.bart_score.score(
            references, predictions, batch_size=batch_size
        )

        # hypo to ref scores are the recall
        hypo_ref_scores = self.bart_score.score(
            predictions, references, batch_size=batch_size
        )

        # calculate max and average for each pair in the batch
        max_avg_f = [
            0.5 * (ref + hypo) for ref, hypo in zip(ref_hypo_scores, hypo_ref_scores)
        ]

        return {
            "f1": max_avg_f,
            "precision": ref_hypo_scores,
            "recall": hypo_ref_scores,
        }
