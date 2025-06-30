import math
import random

from sentence_transformers.losses import ContrastiveTensionDataLoader

from input_example import InputExampleWithGraph


class ContrastiveTensionExampleDataLoader(ContrastiveTensionDataLoader):
    def __init__(
        self,
        examples,
        batch_size,
        pos_neg_ratio=8,
    ):
        super().__init__(
            sentences=examples, batch_size=batch_size, pos_neg_ratio=pos_neg_ratio
        )

    def __iter__(self):
        random.shuffle(self.sentences)
        example_idx = 0
        batch = []

        while example_idx + 1 < len(self.sentences):
            if len(batch) % self.pos_neg_ratio > 0:  # Negative (different) pair
                example_idx += 1
                s2 = self.sentences[example_idx]
                label = 0
                s1 = self.sentences[
                    example_idx - 1
                ]  # For negative pair, get previous example
                if s1.edge_index:
                    batch.append(
                        InputExampleWithGraph(
                            texts=[s1.texts[0], s2.texts[0]],
                            label=label,
                            edge_index=[s1.edge_index[0], s2.edge_index[0]],
                            edge_type=[s1.edge_type[0], s2.edge_type[0]],
                            pos_ids=[s1.pos_ids[0], s2.pos_ids[0]],
                        )
                    )
                else:
                    batch.append(
                        InputExampleWithGraph(
                            texts=[s1.texts[0], s2.texts[0]], label=label
                        )
                    )

                if len(batch) >= self.batch_size:
                    yield (
                        self.collate_fn(batch) if self.collate_fn is not None else batch
                    )
                    batch = []
                    example_idx += 1

            else:  # Positive (identical pair)
                s1 = self.sentences[example_idx]
                label = 1
                s1.set_label(label)
                batch.append(s1)

                if len(batch) >= self.batch_size:
                    yield (
                        self.collate_fn(batch) if self.collate_fn is not None else batch
                    )
                    batch = []
                    example_idx += 1

    def __len__(self):
        return math.floor(len(self.sentences) / (2 * self.batch_size))
