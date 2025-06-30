# %%
import argparse
import json
import logging

from sentence_transformers import LoggingHandler
from sentence_transformers.evaluation import SimilarityFunction

from evaluator import MyEmbeddingSimilarityEvaluator
from input_example import InputExampleWithGraph
from preprocess import generate_ref_edge
from trainer import SentenceTransformerWithGraphs

args = argparse.Namespace(
    dataset_path="data/sick/main/test.json",
    model_path="output/qqp-finetune-pam",
)

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

model = SentenceTransformerWithGraphs(args.model_path)
tokenizer = model.tokenizer
test_samples = []
with open(args.dataset_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        score = float(line["score"])
        max_seq_length = model.max_seq_length
        edge_index, edge_type, pos_ids = generate_ref_edge(
            line, tokenizer, max_seq_length
        )
        inp_example = InputExampleWithGraph(
            texts=[line["graph_ref1"]["amr_simple"], line["graph_ref2"]["amr_simple"]],
            label=score,
            edge_index=edge_index,
            edge_type=edge_type,
            pos_ids=pos_ids,
        )
        test_samples.append(inp_example)
test_evaluator = MyEmbeddingSimilarityEvaluator.from_input_examples(
    test_samples, name="test", batch_size=128
)

parts = args.dataset_path.split("/")
dataset = parts[1]
data_type = parts[2]
print(f"Dataset: {dataset}")
print(f"Type: {data_type}")
test_evaluator.main_similarity = SimilarityFunction.COSINE
pearson_score, cosine_scores = test_evaluator(model)

# %%
import evaluate

sbert = evaluate.load("metrics/sbert")

predictions, references = [], []
for line in lines:
    line = eval(line)
    predictions.append(line["ref1"])
    references.append(line["ref2"])

sbert_scores = sbert.compute(predictions=predictions, references=references)["scores"]

# %%
import matplotlib.pyplot as plt
import numpy as np

min_val = min(min(cosine_scores), min(sbert_scores))
max_val = max(max(cosine_scores), max(sbert_scores))
bins = np.linspace(min_val, max_val, 30)

plt.figure(figsize=(10, 6))

# Plot cosine_scores
plt.hist(
    cosine_scores,
    bins=bins,
    # color="skyblue",
    edgecolor="black",
    alpha=0.7,
    label="PAM",
)

# Plot sbert_scores
plt.hist(
    sbert_scores,
    bins=bins,
    # color="orange",
    edgecolor="black",
    alpha=0.7,
    label="SBERT",
)

# Adding titles and labels
# plt.title('Distribution of Cosine Similarity Scores', fontsize=16)
# plt.xlabel('Cosine Similarity', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.legend(fontsize=14)

plt.show()

# %%
