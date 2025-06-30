import argparse
import json
import logging

import evaluate
from absl import logging as absl_logging
from alignscore import AlignScore
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Set the root logger level to INFO to suppress debug-level messages
logging.basicConfig(level=logging.INFO)

# Set absl logging to WARNING to suppress info messages like "Using default tokenizer."
absl_logging.set_verbosity(absl_logging.WARNING)


def compute_correlations(labels, metric_scores):
    pearson_corr, _ = pearsonr(labels, metric_scores)
    spearman_corr, _ = spearmanr(labels, metric_scores)

    return {"pearson": pearson_corr, "spearman": spearman_corr}


parser = argparse.ArgumentParser(description="Evaluate metrics.")
parser.add_argument(
    "--dataset_path",
    type=str,
    default="data/stsb/main/test.json",
    help="Path to the test dataset file (JSON format).",
)
parser.add_argument(
    "--metric",
    type=str,
    choices=[
        "sbert",
        "bertscore",
        "bartscore",
        "bleu",
        "rouge1",
        "rouge2",
        "rougeL",
        "meteor",
        "alignscore",
    ],
    help="Metric to evaluate (e.g., 'sbert', 'bertscore', 'bleu', 'rouge1').",
)

args = parser.parse_args()


labels = []
hypotheses = []
references = []

with open(args.dataset_path, "r") as f:
    for line in f:
        line = json.loads(line)
        labels.append(float(line["score"]))
        hypotheses.append(line["graph_ref2"]["amr_simple"])
        references.append(line["graph_ref1"]["amr_simple"])

if args.metric in ["bleu", "meteor", "rouge1", "rouge2", "rougeL"]:
    if args.metric in ["rouge1", "rouge2", "rougeL"]:
        m = evaluate.load("rouge")
    else:
        m = evaluate.load(args.metric)
    metric_scores = [
        m.compute(predictions=[pred], references=[[ref]])[args.metric]
        for pred, ref in tqdm(
            zip(hypotheses, references), total=len(hypotheses), desc=args.metric
        )
    ]
elif args.metric == "bertscore":
    bertscore = evaluate.load(args.metric)
    metric_scores = bertscore.compute(
        predictions=hypotheses, references=references, lang="en"
    )["f1"]
elif args.metric == "bartscore":
    bartscore = evaluate.load("metrics/bartscore")
    metric_scores = bartscore.compute(predictions=hypotheses, references=references)[
        "f1"
    ]
elif args.metric == "sbert":
    sbert = evaluate.load("metrics/sbert")
    metric_scores = sbert.compute(predictions=hypotheses, references=references)[
        "scores"
    ]
elif args.metric == "alignscore":
    scorer = AlignScore(
        model="roberta-base",
        batch_size=32,
        device="cuda:0",
        ckpt_path="AlignScore/AlignScore-base.ckpt",
        evaluation_mode="nli_sp",
    )
    metric_scores = scorer.score(contexts=references, claims=hypotheses)
else:
    raise ValueError(f"Unsupported metric: {args.metric}")

correlations = compute_correlations(labels, metric_scores)
print(f"Metric: {args.metric}")
print(f"Pearson: {correlations['pearson']:.4f}")
print(f"Spearman: {correlations['spearman']:.4f}")
