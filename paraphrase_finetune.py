import argparse
import json
import logging
import os

import amrlib
import torch
from datasets import load_dataset
from sentence_transformers import LoggingHandler
from sentence_transformers.losses import OnlineContrastiveLoss
from tqdm import tqdm

import amr_utils
from contrastive_dataloader import ContrastiveExampleDataLoader
from evaluator import MyEmbeddingSimilarityEvaluator
from input_example import InputExampleWithGraph
from preprocess import generate_ref_edge
from trainer import SentenceTransformerWithGraphs

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

logging.getLogger("penman.layout").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_path",
    type=str,
    default="output/pam",  # Path to the pre-trained model
    help="Path to the pre-trained model for further fine-tuning",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default=None,
    help="Path to the processed dataset",
)
parser.add_argument(
    "--batch_size", type=int, default=16, help="Batch size for training"
)
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument(
    "--evaluation_steps",
    type=int,
    default=1000,
    help="Number of steps between evaluations",
)
parser.add_argument(
    "--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer"
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=1000,
    help="Number of warmup steps for the learning rate scheduler",
)
parser.add_argument(
    "--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer"
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=128,
    help="Maximum sequence length for the model",
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="AdamW",
    help="Name of the optimizer to use (e.g., Adam, AdamW, SGD)",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="output/pam-qqp-finetuned",  # Path to save the fine-tuned model
    help="Path to save the fine-tuned model",
)
args = parser.parse_args()

print("Running configuration:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")


def load_qqp_with_amr(split, cache_dir="cached_data"):
    """
    Loads QQP dataset and extracts AMR graphs for each sentence efficiently.

    Args:
        split (str): The dataset split to load ("train" or "validation").
        batch_size (int): Number of sentences to process in a single batch.
        cache_dir (str): Directory to save/load cached processed data.

    Returns:
        List[dict]: A list of dictionaries containing the original sentences and their AMR graphs.
    """
    # File path for cached data
    cache_file = os.path.join(cache_dir, f"qqp_{split}_with_amr.pt")

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Check if cached data exists
    if os.path.exists(cache_file):
        logging.info(f"Loading cached data from {cache_file}")
        return torch.load(cache_file)

    logging.info(f"Processing QQP dataset ({split}) and extracting AMR graphs...")

    # Load the QQP dataset
    dataset = load_dataset(
        "sentence-transformers/quora-duplicates", "triplet", split=split
    )

    # Load the AMR parser
    stog = amrlib.load_stog_model(
        model_dir="amr_parser",
        device="cuda:0",
        batch_size=3,
    )

    processed_data = []
    pbar = tqdm(dataset, desc="Processing dataset")
    for entry in pbar:
        try:
            # Parse the three related sentences as a batch
            batch = [entry["anchor"], entry["positive"], entry["negative"]]
            amr_graphs = stog.parse_sents(batch)

            # Validate the result
            if (
                not amr_graphs
                or len(amr_graphs) != 3
                or any(g is None for g in amr_graphs)
            ):
                logging.warning(f"Skipping entry due to AMR parsing issue: {entry}")
                continue

            # Store the results
            processed_data.append(
                {
                    "anchor": entry["anchor"],
                    "anchor_amr": amr_graphs[0],
                    "positive": entry["positive"],
                    "positive_amr": amr_graphs[1],
                    "negative": entry["negative"],
                    "negative_amr": amr_graphs[2],
                }
            )

            # Update the progress bar with the size of processed_data
            pbar.set_description(f"Processed: {len(processed_data)}")

        except Exception as e:
            logging.warning(f"Failed to process entry: {entry} | Error: {e}")

    # Save processed data to the cache
    logging.info(f"Saving processed data to {cache_file}")
    torch.save(processed_data, cache_file)

    return processed_data


# Load the training and development data
train_samples = load_qqp_with_amr("train", cache_dir="cached_data")


def process_amr_in_dataset(dataset, dataset_path):
    if os.path.exists(dataset_path):
        logging.info(f"Loading processed data from {dataset_path}")
        return torch.load(dataset_path)

    processed_data = []

    for idx, sample in enumerate(dataset):
        # Process anchor
        anchor_amr_entry = sample["anchor_amr"].split("\n", 1)[1]
        anchor_result = amr_utils.convert_amr_to_graph(anchor_amr_entry)
        if anchor_result is None:
            print(f"Skipping entry {idx}: Failed to process anchor AMR.")
            continue
        anchor_tokens, anchor_triples = anchor_result

        # Process positive example
        positive_amr_entry = sample["positive_amr"].split("\n", 1)[1]
        positive_result = amr_utils.convert_amr_to_graph(positive_amr_entry)
        if positive_result is None:
            print(f"Skipping entry {idx}: Failed to process positive AMR.")
            continue
        positive_tokens, positive_triples = positive_result

        # Process negative example
        negative_amr_entry = sample["negative_amr"].split("\n", 1)[1]
        negative_result = amr_utils.convert_amr_to_graph(negative_amr_entry)
        if negative_result is None:
            print(f"Skipping entry {idx}: Failed to process negative AMR.")
            continue
        negative_tokens, negative_triples = negative_result

        # Compile the result for this entry
        processed_entry = {
            "id": str(idx),
            "anchor_tokens": " ".join(anchor_tokens),
            "anchor_triples": anchor_triples,
            "positive_tokens": " ".join(positive_tokens),
            "positive_triples": positive_triples,
            "negative_tokens": " ".join(negative_tokens),
            "negative_triples": negative_triples,
        }

        processed_data.append(processed_entry)

    logging.info(f"Saving processed data to {dataset_path}")
    torch.save(processed_data, dataset_path)

    return processed_data


train_samples = process_amr_in_dataset(train_samples, dataset_path=args.dataset_path)

train_dataloader = ContrastiveExampleDataLoader(
    train_samples,
    batch_size=args.batch_size,
    max_seq_length=args.max_seq_length,
)

# Load the pre-trained model
model = SentenceTransformerWithGraphs(args.model_path)
tokenizer = model._first_module().tokenizer

dev_sts_dataset_path = "data/sick/main/dev.json"  # "data/stsb/main/dev.json"
dev_samples = []

with open(dev_sts_dataset_path, "r") as f:
    lines = f.readlines()
    for line in tqdm(lines, desc="Processing dev data"):
        line = json.loads(line)
        score = float(line["score"])
        edge_index, edge_type, pos_ids = generate_ref_edge(
            line, tokenizer, args.max_seq_length
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
        dev_samples.append(inp_example)
dev_evaluator = MyEmbeddingSimilarityEvaluator.from_input_examples(
    dev_samples, name="sts-dev"
)

# Define loss function
train_loss = OnlineContrastiveLoss(model)

# Fine-tuning the model
logging.info("Starting fine-tuning on QQP dataset...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=args.epochs,
    evaluation_steps=args.evaluation_steps,
    weight_decay=args.weight_decay,
    warmup_steps=args.warmup_steps,
    optimizer_params={"lr": args.learning_rate},
    output_path=args.output_path,
    use_amp=False,
)

logging.info("Fine-tuning completed!")
