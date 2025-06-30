import smatch
from amrlib.evaluate.smatch_score import corpus_bleu as compute_semb
import argparse

def read_amr_file(p1):
    with open(p1, "r") as f:
        amrs = f.read().split("\n\n")
    amrs = [amr.split("#")[-1].split("\n", 1)[1] for amr in amrs]
    return amrs

parser = argparse.ArgumentParser(description="Evaluate metrics.")
parser.add_argument(
    "--src_file_path",
    type=str,
    default="data/stsb/main/raw/src.test.amr",
    help="Path to the source AMR graphs.",
)
    "--tgt_file_path",
    type=str,
    default="data/stsb/main/raw/tgt.test.amr",
    help="Path to the target AMR graphs.",
)
parser.add_argument(
    "--metric",
    type=str,
    choices=[
        "smatch",
        "sembleu",
    ],
    help="Metric to evaluate (e.g., 'smatch', 'sembleu').",
)

args = parser.parse_args()



if args.metric == "smatch":

    def compute_smatch_score(amr1, amr2):
        smatch.reset()
        smatch.parse_amr_line(amr1, 1)
        smatch.parse_amr_line(amr2, 2)
        precision, recall, f1 = smatch.get_f()
        return precision, recall, f1

    smatch_precision, smatch_recall, smatch_f1 = compute_smatch_score(amr_1, amr_2)
    print(
        f"SMATCH - Precision: {smatch_precision:.4f}, Recall: {smatch_recall:.4f}, F1: {smatch_f1:.4f}"
    )
if args.metric == "sembleu":

    def compute_semb_score(amr1, amr2):
        refs = [amr1]  # Reference graph(s)
        hyp = [amr2]  # Hypothesis graph
        return compute_semb(refs, hyp)

    sembleu_score = compute_semb_score(amr_1, amr_2)
    print(f"SEMBLEU - Score: {sembleu_score:.4f}")

