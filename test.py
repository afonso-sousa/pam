import argparse
import json
import logging

from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
)

from evaluator import MyEmbeddingSimilarityEvaluator
from input_example import InputExampleWithGraph
from preprocess import generate_ref_edge
from trainer import SentenceTransformerWithGraphs

parser = argparse.ArgumentParser(description="Evaluate trained models.")
parser.add_argument(
    "--dataset_path",
    type=str,
    default="data/stsb/main/test.json",
    help="Path to the test dataset file (JSON format).",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="output/ct-amr-bert",
    help="Path to model to load.",
)
parser.add_argument(
    "--without_graph", action="store_true", help="Do not add graph information"
)

args = parser.parse_args()

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

if "pam_as_sbert" in args.model_path:
    model = SentenceTransformer(args.model_path)
else:
    model = SentenceTransformerWithGraphs(args.model_path)
tokenizer = model.tokenizer
test_samples = []
with open(args.dataset_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        score = float(line["score"])
        if not args.without_graph:
            max_seq_length = model.max_seq_length
            edge_index, edge_type, pos_ids = generate_ref_edge(
                line, tokenizer, max_seq_length
            )
            inp_example = InputExampleWithGraph(
                texts=[
                    line["graph_ref1"]["amr_simple"],
                    line["graph_ref2"]["amr_simple"],
                ],
                label=score,
                edge_index=edge_index,
                edge_type=edge_type,
                pos_ids=pos_ids,
            )
        else:
            inp_example = InputExample(
                texts=[
                    line["ref1"],
                    line["ref2"],
                ],
                label=score,
            )
        test_samples.append(inp_example)

# if model._first_module().extra_config["arch"] == "pam_as_sbert":
if type(model) is SentenceTransformer:
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        test_samples, name="test", batch_size=128
    )
else:
    test_evaluator = MyEmbeddingSimilarityEvaluator.from_input_examples(
        test_samples, name="test", batch_size=128
    )

parts = args.dataset_path.split("/")
dataset = parts[1]
data_type = parts[2]
print(f"Dataset: {dataset}")
print(f"Type: {data_type}")
test_evaluator.main_similarity = SimilarityFunction.COSINE
_ = test_evaluator(model)
