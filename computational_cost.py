import logging
import time
import warnings

import amrlib
import evaluate
from sentence_transformers import LoggingHandler
from sklearn.metrics.pairwise import paired_cosine_distances

import amr_utils
from preprocess import generate_edge_tensors
from trainer import SentenceTransformerWithGraphs

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="transformers.modeling_utils",
)

logging.getLogger("amrlib").setLevel(logging.ERROR)

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# Load the trained model
model_path = "output/qqp-finetune-pam"
logging.info(f"Loading model from {model_path}")
model = SentenceTransformerWithGraphs(model_path)

tokenizer = model.tokenizer
stog = amrlib.load_stog_model(
    model_dir="amr_parser",
    device="cuda:0",
    batch_size=2,
)

sbert = evaluate.load("metrics/sbert")

# Define a sentence pair for repeated testing
# sentence1 = "Google bought YouTube."
# sentence2 = "Google acquired YouTube."
sentence1 = "Google made a strategic move when it purchased the video-sharing platform YouTube in 2006, reshaping the online media landscape."
sentence2 = "Google expanded its reach by acquiring YouTube, the leading platform for user-generated video content, in a landmark deal."
batch = [sentence1, sentence2]

# Variables to store timings
pam_total_time = 0
pam_amr_extraction_time = 0
sbert_total_time = 0
iterations = 1000

# PAM computational cost assessment
logging.info("Starting PAM computational cost assessment")
for _ in range(iterations):
    start_time = time.time()

    # AMR extraction
    amr_start = time.time()
    amr_graphs = stog.parse_sents(batch)
    amr_end = time.time()
    pam_amr_extraction_time += amr_end - amr_start

    # AMR processing
    s1_graph = amr_utils.convert_amr_to_graph(amr_graphs[0].split("\n", 1)[1])
    s2_graph = amr_utils.convert_amr_to_graph(amr_graphs[1].split("\n", 1)[1])

    s1_tokens, s1_triples = s1_graph
    s2_tokens, s2_triples = s2_graph

    max_seq_length = 128
    s1_edge_index, s1_edge_type, s1_pos_ids = generate_edge_tensors(
        s1_triples, max_seq_length
    )
    s2_edge_index, s2_edge_type, s2_pos_ids = generate_edge_tensors(
        s2_triples, max_seq_length
    )

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

    cosine_score = 1 - paired_cosine_distances(embeddings1, embeddings2)
    pam_total_time += time.time() - start_time

# SBERT computational cost assessment
logging.info("Starting SBERT computational cost assessment")
for _ in range(iterations):
    start_time = time.time()
    sbert_score = sbert.compute(predictions=[sentence1], references=[sentence2])[
        "scores"
    ]
    sbert_total_time += time.time() - start_time

# Results
logging.info("--- Computational Cost Assessment Results ---")
logging.info(
    f"PAM Total Time for {iterations} iterations: {pam_total_time:.2f} seconds"
)
logging.info(
    f"PAM Average Time per iteration: {pam_total_time / iterations:.4f} seconds"
)
logging.info(
    f"PAM AMR Extraction Time for {iterations} iterations: {pam_amr_extraction_time:.2f} seconds"
)
logging.info(
    f"PAM AMR Extraction Average Time per iteration: {pam_amr_extraction_time / iterations:.4f} seconds"
)

logging.info(
    f"SBERT Total Time for {iterations} iterations: {sbert_total_time:.2f} seconds"
)
logging.info(
    f"SBERT Average Time per iteration: {sbert_total_time / iterations:.4f} seconds"
)

metrics = ["bleu", "meteor", "rouge1", "rouge2", "rougeL"]
metrics_times = {}

for metric in metrics:
    logging.info(f"Assessing {metric}...")
    m = evaluate.load("rouge") if metric.startswith("rouge") else evaluate.load(metric)

    for _ in range(iterations):
        start_time = time.time()
        metric_scores = m.compute(predictions=[sentence1], references=[[sentence2]])[
            metric
        ]
        metrics_times[metric] = metrics_times.get(metric, 0) + time.time() - start_time

for metric, total_time in metrics_times.items():
    logging.info(
        f"{metric.upper()} Total Time for {iterations} iterations: {total_time:.2f} seconds"
    )
    logging.info(
        f"{metric.upper()} Average Time per iteration: {total_time / iterations:.4f} seconds"
    )
