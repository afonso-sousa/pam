import json
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from transformers import AutoConfig, AutoTokenizer

from .amrsim import AMRSim
from .glossary import EDGES_AMR
from .graph_model import GraphModel
from .pam import PAM
from .pam_as_sbert import PAMAsSBERT
from .pam_gat import PAMGAT
from .pam_no_ca import PAMNoCA
from .pam_only_last import PAMOnlyLast
from .pam_roberta import PAMRoberta


class GraphSentenceEncoder(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: Optional[int] = None,
        model_args: Dict = {},
        cache_dir: Optional[str] = None,
        tokenizer_args: Dict = {},
        do_lower_case: bool = False,
        tokenizer_name_or_path: str = None,
        gnn_size=128,
        num_gnn_layers=2,
        arch="gnn_only",
        alpha=0.1,
        theta=0.5,
    ):
        super(GraphSentenceEncoder, self).__init__()
        self.config_keys = ["max_seq_length", "do_lower_case"]
        self.do_lower_case = do_lower_case

        config = AutoConfig.from_pretrained(
            model_name_or_path, **model_args, cache_dir=cache_dir
        )
        self.extra_config = {
            "gnn_size": gnn_size,
            "num_gnn_layers": num_gnn_layers,
            "arch": arch,
            "alpha": alpha,
            "theta": theta,
        }

        config.gnn_size = gnn_size
        config.num_gnn_layers = num_gnn_layers
        config.arch = arch
        config.alpha = alpha
        config.theta = theta

        self._load_model(model_name_or_path, config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            (
                tokenizer_name_or_path
                if tokenizer_name_or_path is not None
                else model_name_or_path
            ),
            cache_dir=cache_dir,
            **tokenizer_args,
        )

        # Add AMR edges to tokenizer and update model embeddings
        new_tokens_vocab = {
            "additional_special_tokens": sorted(EDGES_AMR, reverse=True)
        }
        num_added_toks = self.tokenizer.add_special_tokens(new_tokens_vocab)
        print(f"{num_added_toks} tokens added.")
        self.auto_model.resize_token_embeddings(len(self.tokenizer))

        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if (
                hasattr(self.auto_model, "config")
                and hasattr(self.auto_model.config, "max_position_embeddings")
                and hasattr(self.tokenizer, "model_max_length")
            ):
                max_seq_length = min(
                    self.auto_model.config.max_position_embeddings,
                    self.tokenizer.model_max_length,
                )

        self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def _load_model(self, model_name_or_path, config):
        """Loads the transformer model"""
        if config.arch == "gnn_only":
            self.auto_model = GraphModel.from_pretrained(
                model_name_or_path, config=config
            )
        elif config.arch == "amrsim":
            self.auto_model = AMRSim.from_pretrained(model_name_or_path, config=config)
        elif config.arch == "pam":
            self.auto_model = PAM.from_pretrained(model_name_or_path, config=config)
        elif config.arch == "pam_no_ca":
            self.auto_model = PAMNoCA.from_pretrained(model_name_or_path, config=config)
        elif config.arch == "pam_roberta":
            self.auto_model = PAMRoberta.from_pretrained(
                model_name_or_path, config=config
            )
        elif config.arch == "pam_gat":
            self.auto_model = PAMGAT.from_pretrained(model_name_or_path, config=config)
        elif config.arch == "pam_only_last":
            self.auto_model = PAMOnlyLast.from_pretrained(
                model_name_or_path, config=config
            )
        elif config.arch in ["pam_as_sbert", "pam_no_gnn_adapter"]:
            self.auto_model = PAMAsSBERT.from_pretrained(
                model_name_or_path, config=config
            )
        else:
            raise ValueError("Invalid architecture name")

    def __repr__(self):
        return "Transformer({}) with Transformer model: {} ".format(
            self.get_config_dict(), self.auto_model.__class__.__name__
        )

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"],
        }
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]

        # adjust graph edges to batch
        edge_index = features.get("edge_index", None)
        edge_type = features.get("edge_type", None)

        # transform graph into pytorch Geometric format
        if edge_index is not None:
            graph_batch = create_graph_batch(
                features["input_ids"], edge_index, edge_type
            )
            trans_features["graph_batch"] = graph_batch
            trans_features["position_ids"] = features["pos_ids"]

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update(
            {
                "token_embeddings": output_tokens,
                "attention_mask": features["attention_mask"],
            }
        )

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if (
                len(output_states) < 3
            ):  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({"all_layer_embeddings": hidden_states})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(
            self.tokenizer(
                *to_tokenize,
                padding=True,
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.max_seq_length,
            )
        )
        return output

    def get_config_dict(self):
        config_dict = {key: self.__dict__[key] for key in self.config_keys}
        config_dict.update(self.extra_config)
        return config_dict

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, "sentence_bert_config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @classmethod
    def load(cls, input_path: str):
        # Old classes used other config names than 'sentence_bert_config.json'
        for config_name in [
            "sentence_bert_config.json",
            "sentence_roberta_config.json",
            "sentence_distilbert_config.json",
            "sentence_camembert_config.json",
            "sentence_albert_config.json",
            "sentence_xlm-roberta_config.json",
            "sentence_xlnet_config.json",
        ]:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)

        return cls(model_name_or_path=input_path, **config)


def create_graph_batch(embeddings, edge_index, edge_type):
    list_geometric_data = [
        Data(
            x=emb,
            edge_index=edge_index[idx].clone().detach(),
            y=edge_type[idx].clone().detach(),
        )
        for idx, emb in enumerate(embeddings)
    ]

    bdl = Batch.from_data_list(list_geometric_data)
    if torch.cuda.is_available():
        bdl = bdl.to("cuda:" + str(torch.cuda.current_device()))
    else:
        bdl = bdl.to("cpu")

    return bdl
