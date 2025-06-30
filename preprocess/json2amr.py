import argparse
import json
import os
import sys

import penman

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import amr_utils


def ensure_directory_exists(filepath):
    """Ensure the directory for the given filepath exists. Create it if it doesn't."""
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)


def reverse_simplified_amr(graph_simple, triples):
    """
    Reverses the simplified AMR (in graph_simple) and its triples into the full AMR structure.
    """
    nodes = {}
    amr_lines = []

    # Parse the triples to rebuild the node connections
    for triple in triples:
        subj, obj, rel = triple
        subj_node = f"xv{subj}"
        obj_node = f"xv{obj}"

        if subj_node not in nodes:
            nodes[subj_node] = f"{subj_node} / unknown"
        if obj_node not in nodes:
            nodes[obj_node] = f"{obj_node} / unknown"

        # Add the relationship as an AMR line
        amr_lines.append(f"    :{rel} ({obj_node})")

    # Build the AMR graph with proper indentation
    amr_str = "\n".join(
        [f"({node}" + ("".join(amr_lines) if amr_lines else "") + ")" for node in nodes]
    )

    return amr_str


def process_json_to_amr(input_file, out_src_file, out_tgt_file):
    """
    Reads a JSON file and reconstructs AMR graphs into separate source and target AMR files.
    """
    ensure_directory_exists(out_src_file)
    ensure_directory_exists(out_tgt_file)

    with open(input_file, "r", encoding="utf-8") as f:
        json_data = [json.loads(line.strip()) for line in f.readlines()]

    with open(out_src_file, "w", encoding="utf-8") as src_f, open(
        out_tgt_file, "w", encoding="utf-8"
    ) as tgt_f:
        success_count = 0

        for i, entry in enumerate(json_data):
            ref1_graph = entry.get("graph_ref1", {})
            ref2_graph = entry.get("graph_ref2", {})

            # Extract amr_simple and triples from ref1 and ref2
            amr_simple1 = ref1_graph.get("amr_simple")
            amr_simple2 = ref2_graph.get("amr_simple")
            if isinstance(ref1_graph["triples"], str):
                triples_indices1 = json.loads(ref1_graph.get("triples", "[]"))
            else:
                triples_indices1 = ref1_graph["triples"]
            if isinstance(ref2_graph["triples"], str):
                triples_indices2 = json.loads(ref2_graph.get("triples", "[]"))
            else:
                triples_indices2 = ref2_graph["triples"]

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

                try:
                    # Try processing source AMR
                    src_amr_str = penman.encode(graph1)
                    src_amr_str = amr_utils.add_quotes_around_urls(src_amr_str)
                    src_amr_str = amr_utils.add_quotes_around_fractions(src_amr_str)

                    # Try processing target AMR
                    tgt_amr_str = penman.encode(graph2)
                    tgt_amr_str = amr_utils.add_quotes_around_urls(tgt_amr_str)
                    tgt_amr_str = amr_utils.add_quotes_around_fractions(tgt_amr_str)

                    # Write both source and target AMRs if both succeed
                    src_f.write(f"# ::id {success_count}\n")  # Add ID line
                    src_f.write(f"# ::snt {entry['ref1']}\n")
                    src_f.write(f"{src_amr_str}\n\n")

                    tgt_f.write(f"# ::id {success_count}\n")  # Add ID line
                    tgt_f.write(f"# ::snt {entry['ref2']}\n")
                    tgt_f.write(f"{tgt_amr_str}\n\n")

                    success_count += 1
                    print(f"Successfully processed entry {success_count}.")
                except Exception as e:
                    print(
                        f"Error processing entry {i + 1} (ref1: {entry.get('ref1')}, ref2: {entry.get('ref2')}): {e}"
                    )

        print(
            f"Successfully processed {success_count} out of {len(json_data)} entries."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JSON to AMR format for source and target graphs."
    )

    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSON file."
    )
    parser.add_argument(
        "--out_src_file",
        type=str,
        required=True,
        help="Path to the output source AMR file.",
    )
    parser.add_argument(
        "--out_tgt_file",
        type=str,
        required=True,
        help="Path to the output target AMR file.",
    )

    args = parser.parse_args()

    input_file = args.input_file
    out_src_file = args.out_src_file
    out_tgt_file = args.out_tgt_file

    process_json_to_amr(input_file, out_src_file, out_tgt_file)
