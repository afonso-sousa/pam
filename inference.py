import logging

import amrlib
import evaluate
from sentence_transformers import LoggingHandler
from sklearn.metrics.pairwise import paired_cosine_distances
from tqdm import tqdm

import amr_utils
from preprocess import generate_edge_tensors
from trainer import SentenceTransformerWithGraphs

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# Load the trained model
model_path = "output/qqp-finetune-pam"

sentence_pairs = []
####################
# Synonym substitution (positive)
sentence1 = "Google bought YouTube."
sentence2 = "Google acquired YouTube."
sentence_pairs.append((sentence1, sentence2))

# Synonym substitution (negative)
sentence1 = "Google bought YouTube."
sentence2 = "Google destroyed YouTube."
sentence_pairs.append((sentence1, sentence2))

####################

# Antonym substitution (positive)
sentence1 = "Pat ate."
sentence2 = "Pat did not starve."
sentence_pairs.append((sentence1, sentence2))

# Antonym substitution (negative)
sentence1 = "Pat ate."
sentence2 = "Pat did not eat."
sentence_pairs.append((sentence1, sentence2))

####################

# Converse substitution (positive)
sentence1 = "Google bought YouTube."
sentence2 = "YouTube was sold to Google."
sentence_pairs.append((sentence1, sentence2))

# Converse substitution (negative)
sentence1 = "Google bought YouTube."
sentence2 = "Google rented YouTube."
sentence_pairs.append((sentence1, sentence2))

####################
# Change of voice (positive)
sentence1 = "Pat loves Chris."
sentence2 = "Chris is loved by Pat."
sentence_pairs.append((sentence1, sentence2))

# Change of voice (negative)
sentence1 = "Pat loves Chris."
sentence2 = "Chris hates Pat."
sentence_pairs.append((sentence1, sentence2))

####################

# Change of person (positive)
sentence1 = "Pat said, “I like football.”"
sentence2 = "Pat said that he liked football."
sentence_pairs.append((sentence1, sentence2))

# Change of person (negative)
sentence1 = "Pat said, “I like football.”"
sentence2 = "Pat heard, “I like football.”"
sentence_pairs.append((sentence1, sentence2))

####################

# Pronoun/Co-referent substitution (positive)
sentence1 = "Pat likes Chris, because she is smart."
sentence2 = "Pat likes Chris, because Chris is smart."
sentence_pairs.append((sentence1, sentence2))

# Pronoun/Co-referent substitution (negative)
sentence1 = "Pat likes Chris, because she is smart."
sentence2 = "Pat likes Chris, because Pat is smart."
sentence_pairs.append((sentence1, sentence2))

####################

# Repetition/Ellipsis (positive)
sentence1 = "Pat can run fast and Chris can run fast, too."
sentence2 = "Pat can run fast and Chris can, too."
sentence_pairs.append((sentence1, sentence2))

# Repetition/Ellipsis (negative)
sentence1 = "Pat can run fast and Chris can run fast, too."
sentence2 = "Pat can run fast and Chris is slow."
sentence_pairs.append((sentence1, sentence2))

####################

# Function word variations (positive)
sentence1 = "Pat showed a nice demo."
sentence2 = "Pat’s demo was nice."
sentence_pairs.append((sentence1, sentence2))

# Function word variations (negative)
sentence1 = "Pat showed a nice demo."
sentence2 = "Pat’s demo was ugly."
sentence_pairs.append((sentence1, sentence2))

####################

# Actor/Action substitution (positive)
sentence1 = "I dislike rash drivers."
sentence2 = "I dislike rash driving."
sentence_pairs.append((sentence1, sentence2))

# Actor/Action substitution (negative)
sentence1 = "I dislike rash drivers."
sentence2 = "I dislike drivers who follow rules."
sentence_pairs.append((sentence1, sentence2))

logging.info(f"Loading model from {model_path}")
model = SentenceTransformerWithGraphs(model_path)
tokenizer = model.tokenizer

stog = amrlib.load_stog_model(
    model_dir="amr_parser",
    device="cuda:0",
    batch_size=2,
)

sbert = evaluate.load("metrics/sbert")

for sentence1, sentence2 in sentence_pairs:
    batch = [sentence1, sentence2]

    amr_graphs = stog.parse_sents(batch)

    s1_graph = amr_utils.convert_amr_to_graph(amr_graphs[0].split("\n", 1)[1])
    if s1_graph is None:
        print(f"Couldn't process sentence 1: {sentence1}")
        exit()
    s1_tokens, s1_triples = s1_graph
    s2_graph = amr_utils.convert_amr_to_graph(amr_graphs[1].split("\n", 1)[1])
    if s2_graph is None:
        print(f"Couldn't process sentence 2: {sentence2}")
        exit()
    s2_tokens, s2_triples = s2_graph

    max_seq_length = 128
    s1_edge_index, s1_edge_type, s1_pos_ids = generate_edge_tensors(
        s1_triples, max_seq_length
    )
    if s1_edge_type[0] is None:
        print(f"Couldn't process sentence 1: {sentence1}")
        exit()
    s2_edge_index, s2_edge_type, s2_pos_ids = generate_edge_tensors(
        s2_triples, max_seq_length
    )
    if s2_edge_type[0] is None:
        print(f"Couldn't process sentence 1: {sentence1}")
        exit()

    embeddings1 = model.encode(
        [" ".join(s1_tokens)],
        graph_index=[s1_edge_index],
        graph_type=[s1_edge_type],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        pos_ids=[s1_pos_ids],
    )
    embeddings2 = model.encode(
        [" ".join(s2_tokens)],
        graph_index=[s2_edge_index],
        graph_type=[s2_edge_type],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        pos_ids=[s2_pos_ids],
    )

    cosine_score = 1 - (paired_cosine_distances(embeddings1, embeddings2))

    print(sentence1)
    print(sentence2)
    print(f"PAM score: {cosine_score[0]:.4f}")
    print(amr_graphs[0].split("\n", 1)[1])
    print(amr_graphs[1].split("\n", 1)[1])

    sbert_score = sbert.compute(predictions=[sentence1], references=[sentence2])[
        "scores"
    ]
    print(f"SBERT score: {sbert_score[0]:.4f}")
    # print(amr_graphs[0].split("\n", 1)[1])
    # print(amr_graphs[1].split("\n", 1)[1])
