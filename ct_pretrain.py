import argparse
import json
import logging
import os

import torch
from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import ContrastiveTensionDataLoader
from sentence_transformers.models import Pooling
from tqdm import tqdm

from ct_dataloader import ContrastiveTensionExampleDataLoader
from ct_loss import ContrastiveTensionLoss
from evaluator import MyEmbeddingSimilarityEvaluator
from input_example import InputExampleWithGraph
from models.graph_encoder import GraphSentenceEncoder
from models.simple_pooling import SimplePooling
from preprocess import generate_ref_edge, generate_wiki_edge
from trainer import SentenceTransformerWithGraphs

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--arch",
    type=str,
    default="pam",
    help="Name of the model architecture",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="bert-base-uncased",
    help="Name of the pre-trained model",
)
parser.add_argument(
    "--batch_size", type=int, default=16, help="Batch size for training"
)
parser.add_argument(
    "--pos_neg_ratio",
    type=int,
    default=4,
    help="Positive to negative ratio for the batches",
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=128,
    help="Maximum sequence length for the model",
)
parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
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
    "--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer"
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="AdamW",
    help="Name of the optimizer to use (e.g., Adam, AdamW, SGD)",
)
parser.add_argument(
    "--freeze_original",
    action="store_true",
    help="Freeze the original model parameters and only train the newly added ones",
)
parser.add_argument("--gnn_size", type=int, default=128, help="Size of the GNN")
parser.add_argument(
    "--num_gnn_layers", type=int, default=2, help="Number of GNN layers"
)
parser.add_argument("--add_graph", action="store_true", help="Add graph information")
parser.add_argument(
    "--model_save_path",
    type=str,
    default="output/ct-amr-bert",
    help="Path to save the trained model",
)
parser.add_argument(
    "--wikipedia_dataset_path",
    type=str,
    default="data/wiki_train_data.json",
    help="Path to the Wikipedia dataset",
)

args = parser.parse_args()

###### FOR TESTING ######
# To test the code, we use a small dataset
# args.batch_size = 4
# args.pos_neg_ratio = 2
# args.wikipedia_dataset_path = "data/wiki_data_snippet.json"

print("Running configuration:")
print(f"Architecture: {args.arch}")
print(f"Model Name: {args.model_name}")
print(f"Batch Size: {args.batch_size}")
print(f"Positive-Negative Ratio: {args.pos_neg_ratio}")
print(f"Max Sequence Length: {args.max_seq_length}")
print(f"Epochs: {args.epochs}")
print(f"Evaluation Steps: {args.evaluation_steps}")
print(f"Weight Decay: {args.weight_decay}")
print(f"Warmup Steps: {args.warmup_steps}")
print(f"Learning Rate: {args.learning_rate}")
print(f"Optimizer: {args.optimizer}")
print(f"GNN Size: {args.gnn_size}")
print(f"Num GNN Layers: {args.num_gnn_layers}")
print(f"Add Graph Information: {args.add_graph}")
print(f"Model Save Path: {args.model_save_path}")
print(f"Wikipedia Dataset Path: {args.wikipedia_dataset_path}")


def load_or_process_data(dataset_path, processed_data_path, add_graph, max_seq_length):
    if os.path.exists(processed_data_path):
        logging.info(f"Loading processed data from {processed_data_path}")
        return torch.load(processed_data_path)
    else:
        logging.info("Processed data not found. Starting data processing...")
        train_samples = []
        with open(dataset_path, "r", encoding="utf8") as fIn:
            lines = fIn.readlines()
            for line in tqdm(lines, desc="Processing train data"):
                line = json.loads(line)
                if add_graph:
                    graph_triples = line["aligned_triples"]
                    if not graph_triples:
                        continue
                    edge_index, edge_type, pos_ids = generate_wiki_edge(
                        graph_triples, max_seq_length
                    )
                    if edge_index[0] is None:
                        continue
                    inp_example = InputExampleWithGraph(
                        texts=[line["amr_simple"], line["amr_simple"]],
                        edge_index=edge_index,
                        edge_type=edge_type,
                        pos_ids=pos_ids,
                    )
                    train_samples.append(inp_example)
                else:
                    train_samples.append(line["tokens"])

        # Save processed data to disk
        logging.info(f"Saving processed data to {processed_data_path}")
        torch.save(train_samples, processed_data_path)
        return train_samples


processed_data_path = f"processed_{os.path.splitext(os.path.basename(args.wikipedia_dataset_path))[0]}_{'with_graph' if args.add_graph else 'without_graph'}.pt"
cache_dir = "cached_data"
os.makedirs(cache_dir, exist_ok=True)
processed_data_path = os.path.join(cache_dir, processed_data_path)

# Use the function to load or process the data
train_samples = load_or_process_data(
    args.wikipedia_dataset_path,
    processed_data_path,
    add_graph=args.add_graph,
    max_seq_length=args.max_seq_length,
)

if args.arch == "pam_as_sbert":
    train_dataloader = ContrastiveTensionDataLoader(
        train_samples,
        batch_size=args.batch_size,
        pos_neg_ratio=args.pos_neg_ratio,
    )
else:
    train_dataloader = ContrastiveTensionExampleDataLoader(
        train_samples,
        batch_size=args.batch_size,
        pos_neg_ratio=args.pos_neg_ratio,
    )

################# Intialize an SBERT model #################
word_embedding_model = GraphSentenceEncoder(
    args.model_name,
    max_seq_length=args.max_seq_length,
    gnn_size=args.gnn_size,
    num_gnn_layers=args.num_gnn_layers,
    arch=args.arch,
)
tokenizer = word_embedding_model.tokenizer

pooling_model = (
    SimplePooling(word_embedding_model.get_word_embedding_dimension())
    if args.arch == "gnn_only"
    else Pooling(word_embedding_model.get_word_embedding_dimension())
)

if args.arch == "pam_as_sbert":
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
else:
    model = SentenceTransformerWithGraphs(modules=[word_embedding_model, pooling_model])

# Freeze original parameters if specified
if args.freeze_original:
    for name, param in model.named_parameters():
        if "adapter" not in name and "graph" not in name:
            param.requires_grad = False
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Number of trainable parameters: {trainable_params:_}")
logging.info(f"Total number of parameters: {total_params:_}")

dev_sts_dataset_path = "data/sick/main/dev.json"  # "data/stsb/main/dev.json"
dev_samples = []

with open(dev_sts_dataset_path, "r") as f:
    lines = f.readlines()
    for line in tqdm(lines, desc="Processing dev data"):
        line = json.loads(line)
        score = float(line["score"])
        if args.add_graph:
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
        else:
            inp_example = InputExample(
                texts=[
                    line["ref1"],
                    line["ref2"],
                ],
                label=score,
            )
        dev_samples.append(inp_example)

if args.arch == "pam_as_sbert":
    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dev_samples, name="sts-dev"
    )
else:
    dev_evaluator = MyEmbeddingSimilarityEvaluator.from_input_examples(
        dev_samples, name="sts-dev"
    )

################# Train an SBERT model #################

train_loss = ContrastiveTensionLoss(model)

optimizer_class = getattr(torch.optim, args.optimizer)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=args.epochs,
    evaluation_steps=args.evaluation_steps,
    weight_decay=args.weight_decay,
    warmup_steps=args.warmup_steps,
    optimizer_class=optimizer_class,
    optimizer_params={"lr": args.learning_rate},
    output_path=args.model_save_path,
    use_amp=False,  # Set to True, if your GPU has optimized FP16 cores
    # param_specific_lr=(
    #     {
    #         "adapter": 2e-4,
    #         "graph_cross_attention": 2e-4,
    #         "graph_projection": 2e-4,
    #     }
    #     if args.arch == "pam4"
    #     else None
    # ),
)
