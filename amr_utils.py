"""
This file has utils functions to extract an AMR graph from a sentence.
It is based on the code from https://github.com/zzshou/AMRSim/blob/main/preprocess/utils.py
"""

import logging
import re

import penman

# Suppress INFO, DEBUG and WARNING logs from penman
logging.getLogger("penman.layout").setLevel(logging.ERROR)


def add_quotes_around_urls(text):
    # Regular expression pattern to match URLs
    url_pattern = r"(https?://[^\s()]+)"

    # Function to add quotes around the matched URL
    return re.sub(url_pattern, r'"\1"', text)


def add_quotes_around_fractions(text):
    # Regular expression pattern to match fractions like 1/2, 1/3, etc.
    fraction_pattern = r"\b(\d+/\d+)\b"

    # Function to add quotes around the matched fraction
    return re.sub(fraction_pattern, r'"\1"', text)


def simplify_amr_tokens(tokens, v2c, out_graph_type="bipartite"):
    """
    Simplifies the linearized AMR by processing the tokens and mapping variables to concepts.

    Args:
        tokens (list): Linearized AMR tokens.
        v2c (dict): A mapping of variables to their corresponding concepts.

    Returns:
        tuple: A list of simplified tokens and a mapping of variables to their token positions.
    """
    mapping = {}  # Mapping from variable names to positions in the token list
    new_tokens = []  # Simplified token list
    save_map = (
        False  # Indicates when to save the mapping from variable to token position
    )
    last_map = None  # Stores the last variable to map
    for tok in tokens:
        # ignore instance-of
        if tok.startswith("("):  # Ignore variable names in tokens
            last_map = tok.replace("(", "")
            continue
        elif tok == "/":  # Indicates that the following token is a concept
            save_map = True
            continue
        elif out_graph_type == "multirelational" and tok.startswith(":"):
            continue  # Ignore relations in multirelational graphs
        # predicates, we remove any alignment information and parenthesis
        elif tok.startswith(":"):  # Process relations (e.g., :ARG0)
            new_tok = tok.strip(")").split("~")[0]
            new_tokens.append(new_tok)
        else:  # Process concepts and reentrancies
            new_tok = tok.strip(")").split("~")[0]

            if new_tok == "":
                continue

            # now we check if it is a concept or a variable (reentrancy)
            if new_tok in v2c:  # If it's a reentrant variable
                if new_tok not in mapping:
                    mapping[new_tok] = set()
                mapping[new_tok].add(len(new_tokens))
                if v2c[new_tok] is not None:
                    new_tok = v2c[new_tok]

            elif new_tok.isnumeric():  # If it's a number, keep it as is
                new_tok = new_tok

            # remove quotes
            elif new_tok[0] == '"' and new_tok[-1] == '"':  # Remove quotes from strings
                new_tok = new_tok[1:-1]

            if new_tok != "":
                new_tokens.append(new_tok)

            if save_map:  # Save the variable-to-position mapping
                if last_map not in mapping:
                    mapping[last_map] = set()
                mapping[last_map].add(len(new_tokens) - 1)
                save_map = False

    return new_tokens, mapping


def find_token_positions(new_tokens, src):
    """
    Retrieves the positions of a specific token in the list of tokens.

    Args:
        new_tokens (list): Simplified tokens.
        src (str): The token to search for.

    Returns:
        list: A list of positions where the token appears.
    """
    pos = []
    for idx, n in enumerate(new_tokens):
        if n == src:
            pos.append(idx)
    return pos


def generate_bipartite_graph(graph, new_tokens, mapping, roles_in_order):
    """
    Converts the AMR graph into a set of bipartite edge triples and node tokens.

    Args:
        graph (penman.Graph): The decoded AMR graph.
        new_tokens (list): Simplified tokens.
        mapping (dict): Mapping of variables to token positions.
        roles_in_order (list): List of roles (relations) in the order they appear.

    Returns:
        tuple: Node tokens and bipartite edge triples.
    """
    try:
        triples = []  # Stores the final edge triples
        nodes_to_print = new_tokens  # Final node tokens

        graph_triples = graph.triples  # Extracted triples from the graph

        edge_id = -1  # Initialize edge identifier
        triples_set = set()  # To avoid duplicate triples
        count_roles = 0  # Index to iterate through roles
        for triple in graph_triples:
            src, edge, tgt = triple
            if edge == ":instance" or edge == ":instance-of":
                continue

            # if penman.layout.appears_inverted(graph_penman, v):
            # Handle "-of" inversions for certain roles
            if (
                "-of" in roles_in_order[count_roles]
                and "-off" not in roles_in_order[count_roles]
            ):
                if edge != ":consist-of":
                    edge = edge + "-of"
                    old_tgt = tgt
                    tgt = src
                    src = old_tgt

            assert roles_in_order[count_roles] == edge, f"Erroneous graph:\n{graph}"
            count_roles += 1

            if edge == ":wiki":
                continue

            src = str(src).replace('"', "")
            tgt = str(tgt).replace('"', "")

            if src not in mapping:
                src_id = find_token_positions(new_tokens, src)
            else:
                src_id = sorted(list(mapping[src]))

            edge_id = find_edge_position(new_tokens, edge, edge_id)

            if tgt not in mapping:
                tgt_id = find_token_positions(new_tokens, tgt)
            else:
                tgt_id = sorted(list(mapping[tgt]))

            for s_id in src_id:
                if (s_id, edge_id, "d") not in triples_set:
                    triples.append((s_id, edge_id, "d"))
                    triples_set.add((s_id, edge_id, "d"))
                    triples.append((edge_id, s_id, "r"))
            for t_id in tgt_id:
                if (edge_id, t_id, "d") not in triples_set:
                    triples.append((edge_id, t_id, "d"))
                    triples_set.add((edge_id, t_id, "d"))
                    triples.append((t_id, edge_id, "r"))

        if nodes_to_print == []:
            # single node graph, first triple is ":top", second triple is the node
            triples.append((0, 0, "s"))
        return nodes_to_print, triples
    except Exception as e:
        print(f"Error processing graph: {e}\nGraph: {graph}")
        return None


def find_edge_position(tokens, edge, edge_id):
    """
    Finds the position of an edge (relation) in the token list.

    Args:
        tokens (list): Simplified tokens.
        edge (str): The edge (relation) to find.
        edge_id (int): Current edge identifier.

    Returns:
        int: The position of the edge in the token list.
    """
    for idx in range(edge_id + 1, len(tokens)):
        if tokens[idx] == edge:
            return idx


def map_variables_to_concepts(graph_penman):
    """
    Creates a dictionary mapping from variables to their corresponding concepts in the AMR graph.

    Args:
        graph_penman (penman.Graph): The decoded AMR graph.

    Returns:
        dict: A mapping of variables to their concepts.
    """
    instances = graph_penman.instances()
    dict_insts = {}
    for i in instances:
        dict_insts[i.source] = i.target
    return dict_insts


def convert_amr_to_graph(amr_str, out_graph_type="bipartite", rel_glossary=None):
    """
    Simplifies an AMR string into node tokens and bipartite edge triples.

    Args:
        amr_str (str): The AMR string.

    Returns:
        tuple: Simplified node tokens and bipartite edge triples.
    """
    amr_str = add_quotes_around_urls(amr_str)
    # Decode AMR string into a Penman graph
    try:
        graph_penman = penman.decode(amr_str)
    except:
        print(f"Error decoding AMR:\n{amr_str}")
        return None
    v2c_penman = map_variables_to_concepts(graph_penman)

    # Linearize and clean up the AMR
    linearized_amr = penman.encode(graph_penman).replace("\t", "").replace("\n", "")
    tokens = linearized_amr.split()

    # Simplify the tokens
    new_tokens, mapping = simplify_amr_tokens(tokens, v2c_penman, out_graph_type)

    # Extract roles in the order they appear
    roles_in_order = [
        token for token in tokens if token.startswith(":") and token != ":instance-of"
    ]

    # Generate node tokens and edge triples
    if out_graph_type == "bipartite":
        out = generate_bipartite_graph(
            graph_penman, new_tokens, mapping, roles_in_order
        )
        return out
    elif out_graph_type == "multirelational":
        nodes, triples, edge_types = generate_multirelational_graph(
            graph_penman, new_tokens, mapping, roles_in_order, rel_glossary
        )
        return nodes, triples, edge_types
    else:
        raise ValueError("Invalid graph type: {}".format(out_graph_type))


def generate_multirelational_graph(
    graph, new_tokens, mapping, roles_in_order, rel_glossary
):
    """
    Converts the AMR graph into a set of multirelational edge triples and node tokens.

    Args:
        graph (penman.Graph): The decoded AMR graph.
        new_tokens (list): Simplified tokens.
        mapping (dict): Mapping of variables to token positions.
        roles_in_order (list): List of roles (relations) in the order they appear.

    Returns:
        tuple: Node tokens and multirelational edge triples.
    """
    assert (
        rel_glossary is not None
    ), "Relation glossary is required for multirelational graphs"

    triples = []  # Stores the final multirelational edge triples
    new_tokens = [
        token for token in new_tokens if not token.startswith(":")
    ]  # node tokens without relations

    graph_triples = graph.triples  # Extracted triples from the graph
    count_roles = 0  # Index to iterate through roles
    edge_types = []  # Stores the edge types

    for triple in graph_triples:
        src, edge, tgt = triple
        if edge == ":instance" or edge == ":instance-of":
            continue

        # Handle "-of" inversions for certain roles
        if (
            "-of" in roles_in_order[count_roles]
            and "-off" not in roles_in_order[count_roles]
        ):
            if edge != ":consist-of":
                edge = edge + "-of"
                src, tgt = tgt, src

        assert roles_in_order[count_roles] == edge
        count_roles += 1

        if edge in [":wiki", ":"]:
            continue

        src = str(src).replace('"', "")
        tgt = str(tgt).replace('"', "")

        # Get the position(s) of the source and target nodes
        src_ids = (
            find_token_positions(new_tokens, src)
            if src not in mapping
            else sorted(list(mapping[src]))
        )
        tgt_ids = (
            find_token_positions(new_tokens, tgt)
            if tgt not in mapping
            else sorted(list(mapping[tgt]))
        )

        try:
            # Get the id of the edge (relation)
            edge_type = rel_glossary.index(edge)
        except:
            breakpoint()

        # Generate multirelational triples (src, edge, tgt)
        for s_id in src_ids:
            for t_id in tgt_ids:
                triples.append((s_id, t_id, 0))
                edge_types.append(edge_type)

    if not new_tokens:
        # Handle single-node graphs (if applicable)
        triples.append((0, 0, 0))
        edge_types.append(0)

    assert len(edge_types) == len(triples)

    return new_tokens, triples, edge_types


def construct_triples(tokens, triples_indices):
    processed_triples = []
    used_indices = set()

    # Map variables to unique labels
    variable_to_label = {}  # TODO: handle repeated tokens
    ascii_offset = 97  # ASCII value of 'a'
    letter_count = 0  # Keeps track of which letter to assign to the next variable
    for token in tokens:
        if not token.startswith(":"):  # Treat as a variable (not a relation)
            letter = chr(ascii_offset + (letter_count % 26))
            suffix = letter_count // 26
            variable_to_label[token] = f"{letter}{suffix}" if suffix > 0 else letter
            letter_count += 1

    # Generate :instance triples
    for variable, label in variable_to_label.items():
        processed_triples.append((label, ":instance", variable))

    # Process standard triples
    for i, (s, o, r) in enumerate(triples_indices):
        if r == "d" and tokens[o].startswith(":") and (s, o) not in used_indices:
            for j, (o2, t, r2) in enumerate(triples_indices):
                if i != j and o2 == o and r2 == "d" and (o, t) not in used_indices:
                    # Replace variables with their labels in the constructed triple
                    processed_triples.append(
                        (
                            variable_to_label[tokens[s]],
                            tokens[o],
                            variable_to_label[tokens[t]],
                        )
                    )
                    used_indices.update({(s, o), (o, t)})
                    break
    return processed_triples


if __name__ == "__main__":
    # Example usage
    amr_str = "(p / prospect-02\n      :ARG0 (p2 / person\n            :ARG0-of (e / engineer-01\n                  :ARG1 (m / mechanics)))\n      :ARG1 (a / amr-unknown)\n      :time (f / future))"

    nodes, triples = convert_amr_to_graph(amr_str)

    print(nodes)
    print(triples)
