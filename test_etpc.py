import argparse
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import amrlib
import datasets
import evaluate
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import LoggingHandler, SimilarityFunction

import amr_utils
from evaluator import MyEmbeddingSimilarityEvaluator
from input_example import InputExampleWithGraph
from preprocess import generate_edge_tensors
from trainer import SentenceTransformerWithGraphs

logging.getLogger("amrlib").setLevel(logging.WARNING)


def get_cache_key():
    """Generate a unique key for the current dataset and models"""
    return f"etpc_pam_{args.model_path.replace('/', '_')}"


def save_computed_scores(cosine_scores, sbert_scores, filtered_dataset):
    cache_data = {
        "cosine_scores": cosine_scores,
        "sbert_scores": sbert_scores,
        "filtered_dataset": filtered_dataset,
    }
    cache_file = Path(f"cached_data/score_cache_{get_cache_key()}.pkl")
    with cache_file.open("wb") as f:
        pickle.dump(cache_data, f)
    print(f"Saved computed scores to {cache_file}")


def load_computed_scores():
    cache_file = Path(f"cached_data/score_cache_{get_cache_key()}.pkl")
    if cache_file.exists():
        with cache_file.open("rb") as f:
            data = pickle.load(f)
            print(f"Loaded cached scores from {cache_file}")
            return data["cosine_scores"], data["sbert_scores"], data["filtered_dataset"]
    return None, None, None


args = argparse.Namespace(
    model_path="output/qqp-finetune-pam",
)

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

cosine_scores, sbert_scores, filtered_dataset = load_computed_scores()
if cosine_scores is None or sbert_scores is None:
    # Load dataset
    dataset = datasets.load_dataset("jpwahle/etpc", split="train")

    # dataset = dataset.select(range(10))
    dataset = dataset.train_test_split(test_size=0.25, seed=42)["test"]

    paraphrase_type_scores = defaultdict(list)

    # Load models
    pam_model = SentenceTransformerWithGraphs(args.model_path)

    test_samples = []
    filtered_dataset = []
    cosine_scores = []
    sbert_scores = []

    stog = amrlib.load_stog_model(
        model_dir="amr_parser",
        device="cuda:0",
        batch_size=2,
    )

    for entry in dataset:
        sent1, sent2 = entry["sentence1"], entry["sentence2"]
        paraphrase_types = entry["paraphrase_types"]

        batch = [sent1, sent2]
        amr_graphs = stog.parse_sents(batch)

        try:
            s1_graph = amr_utils.convert_amr_to_graph(amr_graphs[0].split("\n", 1)[1])
            if s1_graph is None:
                logging.warning(f"Couldn't process sentence 1: {sent1}")
                continue  # Skip to next entry

            s2_graph = amr_utils.convert_amr_to_graph(amr_graphs[1].split("\n", 1)[1])
            if s2_graph is None:
                logging.warning(f"Couldn't process sentence 2: {sent2}")
                continue

            s1_tokens, s1_triples = s1_graph
            s2_tokens, s2_triples = s2_graph

        except Exception as e:
            logging.warning(f"Error processing sentences: {sent1} | {sent2} - {str(e)}")
            continue

        max_seq_length = 128
        s1_edge_index, s1_edge_type, s1_pos_ids = generate_edge_tensors(
            s1_triples, max_seq_length
        )
        if s1_edge_type[0] is None:
            print(f"Couldn't process sentence 1: {sent1}")
            exit()
        s2_edge_index, s2_edge_type, s2_pos_ids = generate_edge_tensors(
            s2_triples, max_seq_length
        )
        if s2_edge_type[0] is None:
            print(f"Couldn't process sentence 1: {sent2}")
            exit()

        inp_example = InputExampleWithGraph(
            texts=[
                " ".join(s1_tokens),
                " ".join(s2_tokens),
            ],
            edge_index=[s1_edge_index, s2_edge_index],
            edge_type=[s1_edge_type, s2_edge_type],
            pos_ids=[s1_pos_ids, s2_pos_ids],
        )
        test_samples.append(inp_example)
        filtered_dataset.append(entry)

    test_evaluator = MyEmbeddingSimilarityEvaluator.from_input_examples(
        test_samples, name="test", batch_size=128
    )
    test_evaluator.main_similarity = SimilarityFunction.COSINE
    _, cosine_scores = test_evaluator(pam_model)

    # Compute SBERT similarity scores
    sbert_metric = evaluate.load("metrics/sbert")
    predictions, references = zip(
        *[(entry["sentence1"], entry["sentence2"]) for entry in filtered_dataset]
    )
    sbert_scores = sbert_metric.compute(predictions=predictions, references=references)[
        "scores"
    ]
    save_computed_scores(cosine_scores, sbert_scores, filtered_dataset)


def z_normalize(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    return (scores - mean) / std


# Normalize PAM and SBERT scores
cosine_scores = z_normalize(np.array(cosine_scores))
sbert_scores = z_normalize(np.array(sbert_scores))

type_pam_scores = defaultdict(list)
type_sbert_scores = defaultdict(list)

for entry, pam_score, sbert_score in zip(filtered_dataset, cosine_scores, sbert_scores):
    for ptype in entry["paraphrase_types"]:
        type_pam_scores[ptype].append(pam_score)
        type_sbert_scores[ptype].append(sbert_score)
    if len(entry["paraphrase_types"]) == 0:
        type_pam_scores["Non-paraphrase"].append(pam_score)
        type_sbert_scores["Non-paraphrase"].append(sbert_score)


# Calculate average scores per type
types = sorted(type_pam_scores.keys())
pam_means = [np.mean(type_pam_scores[t]) for t in types]
sbert_means = [np.mean(type_sbert_scores[t]) for t in types]

# Create the plot
x = np.arange(len(types))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width / 2, pam_means, width, label="PAM", alpha=0.7)
rects2 = ax.bar(x + width / 2, sbert_means, width, label="SBERT", alpha=0.7)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Normalized Similarity Score (Z-score)")
# ax.set_title("Average Similarity Scores by Paraphrase Type")
ax.set_xticks(x)
ax.set_xticklabels(types, rotation=45, ha="right")
ax.legend()


# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height * 100:.0f}",  # f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plot_filename = "paraphrase_type_scores_normalized.png"
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
print(f"Plot saved as {plot_filename}")
plt.show()

###############################
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def evaluate_classification(y_true, scores, model_name):
    # ROC analysis
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Precision-Recall analysis
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)

    return {
        "model": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "optimal_threshold": optimal_threshold,
        "fpr_at_optimal": fpr[optimal_idx],
        "tpr_at_optimal": tpr[optimal_idx],
    }


# Prepare true labels (1 if any paraphrase type exists)
true_labels_binary = np.array(
    [1 if len(entry["paraphrase_types"]) > 0 else 0 for entry in filtered_dataset]
)

# Evaluate both models
pam_results = evaluate_classification(true_labels_binary, cosine_scores, "PAM")
sbert_results = evaluate_classification(true_labels_binary, sbert_scores, "SBERT")


###############################
def evaluate_at_fpr(y_true, scores, target_fpr=0.05):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    idx = np.argmin(np.abs(fpr - target_fpr))
    return {
        "threshold": thresholds[idx],
        "tpr": tpr[idx],
        "fpr": fpr[idx],
        "precision": np.sum((scores >= thresholds[idx]) & (y_true == 1))
        / np.sum(scores >= thresholds[idx]),
    }


pam_fpr5 = evaluate_at_fpr(true_labels_binary, cosine_scores)
sbert_fpr5 = evaluate_at_fpr(true_labels_binary, sbert_scores)

###############################
# Your existing per-type analysis
type_comparison = {
    t: {
        "pam_mean": np.mean(type_pam_scores[t]),
        "sbert_mean": np.mean(type_sbert_scores[t]),
        "delta": np.mean(type_pam_scores[t]) - np.mean(type_sbert_scores[t]),
    }
    for t in types
}

###############################
# ROC Curve comparison
plt.figure(figsize=(10, 6))
for model, scores in [("PAM", cosine_scores), ("SBERT", sbert_scores)]:
    fpr, tpr, _ = roc_curve(true_labels_binary, scores)
    plt.plot(fpr, tpr, label=f"{model} (AUC = {auc(fpr, tpr):.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.savefig("roc_comparison.png", dpi=300, bbox_inches="tight")

###############################
from scipy.stats import ttest_rel

# Paired t-test comparing the scores
t_stat, p_value = ttest_rel(cosine_scores, sbert_scores)
print(f"Paired t-test p-value: {p_value:.4f}")

###############################
print("\n=== Model Comparison Summary ===")
print(f"PAM ROC AUC: {pam_results['roc_auc']:.3f}")
print(f"SBERT ROC AUC: {sbert_results['roc_auc']:.3f}")
print(f"\nAt 5% FPR:")
print(f"PAM TPR: {pam_fpr5['tpr']:.3f}, Precision: {pam_fpr5['precision']:.3f}")
print(f"SBERT TPR: {sbert_fpr5['tpr']:.3f}, Precision: {sbert_fpr5['precision']:.3f}")

print("\nTop 5 types where PAM outperforms SBERT:")
sorted_types = sorted(type_comparison.items(), key=lambda x: -x[1]["delta"])
for t, vals in sorted_types[:5]:
    print(
        f"{t}: Δ={vals['delta']:.3f} (PAM={vals['pam_mean']:.3f}, SBERT={vals['sbert_mean']:.3f})"
    )
