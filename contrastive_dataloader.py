import math
import random

from input_example import InputExampleWithGraph
from preprocess import generate_wiki_edge


class ContrastiveExampleDataLoader:
    def __init__(self, examples, batch_size, max_seq_length):
        self.examples = examples
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.collate_fn = None

    def __iter__(self):
        random.shuffle(self.examples)
        example_idx = 0
        batch = []

        while example_idx + 1 < len(self.examples):
            sample = self.examples[example_idx]

            anchor_edge_index, anchor_edge_type, anchor_pos_ids = generate_wiki_edge(
                sample["anchor_triples"], self.max_seq_length
            )
            if anchor_edge_index[0] is None:
                continue
            positive_edge_index, positive_edge_type, positive_pos_ids = (
                generate_wiki_edge(sample["positive_triples"], self.max_seq_length)
            )
            if positive_edge_index[0] is None:
                continue
            negative_edge_index, negative_edge_type, negative_pos_ids = (
                generate_wiki_edge(sample["negative_triples"], self.max_seq_length)
            )
            if negative_edge_index[0] is None:
                continue

            # Create InputExampleWithGraph objects for anchor-positive and anchor-negative pairs
            anchor_positive = InputExampleWithGraph(
                texts=[sample["anchor_tokens"], sample["positive_tokens"]],
                label=1,  # Positive pair
                edge_index=[anchor_edge_index[0], positive_edge_index[0]],
                edge_type=[anchor_edge_type[0], positive_edge_type[0]],
                pos_ids=[anchor_pos_ids[0], positive_pos_ids[0]],
            )
            anchor_negative = InputExampleWithGraph(
                texts=[sample["anchor_tokens"], sample["negative_tokens"]],
                label=0,  # Negative pair
                edge_index=[anchor_edge_index[0], negative_edge_index[0]],
                edge_type=[anchor_edge_type[0], negative_edge_type[0]],
                pos_ids=[anchor_pos_ids[0], negative_pos_ids[0]],
            )

            # Add pairs to the batch
            batch.extend([anchor_positive, anchor_negative])

            # Check if batch is full
            if len(batch) >= self.batch_size:
                yield self.collate_fn(batch) if self.collate_fn else batch
                batch = []

            example_idx += 1

        # Yield the remaining batch if any
        if batch:
            yield self.collate_fn(batch) if self.collate_fn else batch

    def __len__(self):
        return math.ceil(
            len(self.examples) * 2 / self.batch_size
        )  # Multiply by 2 for pos-neg pairs
