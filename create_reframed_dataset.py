import argparse
import json
import logging
import os
import random

import penman

import amr_utils

# Load dataset from file
parser = argparse.ArgumentParser(description="Create linearization dataset.")
parser.add_argument(
    "--dataset_path",
    type=str,
    default="data/stsb/main/test.json",
    help="Path to the test dataset file (JSON format).",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="data/output/final_dataset.json",
    help="Path to save the final dataset file.",
)

args = parser.parse_args()

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


final_samples = []
with open(args.dataset_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        sample = json.loads(line)
        ref1_graph = sample.get("graph_ref1", {})
        ref2_graph = sample.get("graph_ref2", {})

        # Extract amr_simple and triples from ref1 and ref2
        amr_simple1 = ref1_graph.get("amr_simple")
        amr_simple2 = ref2_graph.get("amr_simple")
        triples_indices1 = json.loads(ref1_graph.get("triples", "[]"))
        triples_indices2 = json.loads(ref2_graph.get("triples", "[]"))

        # Split amr_simple by spaces to get tokens
        tokens1 = amr_simple1.split() if amr_simple1 else []
        tokens2 = amr_simple2.split() if amr_simple2 else []

        # Construct triples based on tokens and triples indices
        triples1 = amr_utils.construct_triples(tokens1, triples_indices1)
        triples2 = amr_utils.construct_triples(tokens2, triples_indices2)

        # Check if we have valid tokens and triples to proceed
        if tokens1 and tokens2 and triples1 and triples2:
            # Create Penman Graphs
            graph1 = penman.Graph(triples1)
            graph2 = penman.Graph(triples2)

            # Generate variants with different top nodes for each graph
            candidate_tops1 = graph1.variables()
            candidate_tops2 = graph2.variables()

            # Encode each graph with different tops to create linearizations
            new_graphs1 = []
            for t in candidate_tops1:
                try:
                    new_graphs1.append(penman.encode(graph1, top=t))
                except Exception as e:
                    print(f"Error encoding graph1 with top '{t}': {e}")

            new_graphs2 = []
            for t in candidate_tops2:
                try:
                    new_graphs2.append(penman.encode(graph2, top=t))
                except Exception as e:
                    print(f"Error encoding graph2 with top '{t}': {e}")

            lin_new_graphs1 = []
            for amr_str in new_graphs1:
                out = amr_utils.convert_amr_to_graph(amr_str)
                if out is not None:
                    lin_new_graphs1.append(list(out))

            lin_new_graphs2 = []
            for amr_str in new_graphs2:
                out = amr_utils.convert_amr_to_graph(amr_str)
                if out is not None:
                    lin_new_graphs2.append(list(out))

            # Sample two or three linearizations from each set
            min_samples = min(len(lin_new_graphs1), len(lin_new_graphs2))
            if min_samples > 0:
                sampled_graphs1 = random.sample(lin_new_graphs1, min_samples)
                sampled_graphs2 = random.sample(lin_new_graphs2, min_samples)

                # Match sampled graphs and add to final samples
                for (tokens1, triples1), (tokens2, triples2) in zip(
                    sampled_graphs1, sampled_graphs2
                ):
                    final_samples.append(
                        {
                            "score": sample["score"],
                            "ref1": sample["ref1"],
                            "ref2": sample["ref2"],
                            "graph_ref1": {
                                "amr_simple": " ".join(tokens1),
                                "triples": list(map(list, triples1)),
                            },
                            "graph_ref2": {
                                "amr_simple": " ".join(tokens2),
                                "triples": list(map(list, triples2)),
                            },
                        }
                    )
            else:
                final_samples.append(
                    {
                        "score": sample["score"],
                        "ref1": sample["ref1"],
                        "ref2": sample["ref2"],
                        "graph_ref1": {
                            "amr_simple": amr_simple1,
                            "triples": triples_indices1,
                        },
                        "graph_ref2": {
                            "amr_simple": amr_simple2,
                            "triples": triples_indices2,
                        },
                    }
                )

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

# Save the final dataset to file
with open(args.output_path, "w") as out_file:
    for sample in final_samples:
        out_file.write(json.dumps(sample) + "\n")

logging.info("Final dataset saved to %s", args.output_path)
