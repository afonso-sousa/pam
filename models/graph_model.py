import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.pool import global_mean_pool
from transformers.models.bert.modeling_bert import BertEncoder, BertPreTrainedModel

from .simple_embeddings import SimpleEmbeddings


class BertMeanPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        mean_tensor = torch.mean(hidden_states, 1)
        pooled_output = self.dense(mean_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GraphModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = SimpleEmbeddings(config)
        self.encoder = GraphBertEncoder(config)

        self.pooler = BertMeanPooler(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        graph_batch=None,
        *args,
        **kwargs,
    ):
        hidden_states = self.embeddings(graph_batch.x)

        encoder_output = self.encoder(
            hidden_states,
            graph_batch,
        )

        pooled_output = self.pooler(encoder_output) if self.pooler is not None else None

        return (encoder_output, pooled_output)


class GraphBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.lins = torch.nn.ModuleList()
        self.lins.append(nn.Linear(config.hidden_size, config.gnn_size))
        self.lins.append(nn.Linear(config.gnn_size, config.hidden_size))
        self.convs = torch.nn.ModuleList()
        for layer in range(config.gnn_layer):
            self.convs.append(
                GCN2Conv(
                    config.gnn_size,
                    config.alpha,
                    config.theta,
                    layer + 1,
                    normalize=False,
                )
            )

        self.dropout = config.hidden_dropout_prob

    def forward(
        self,
        hidden_states,
        graphs,
    ):
        x = F.dropout(hidden_states, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        # [num_nodes, config.intermediate_size]

        _, _, feature_size = x.size()
        x = x.view(-1, feature_size)  # [batch_size * seq_length, gnn_size]
        x_0 = x_0.view(-1, feature_size)

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, graphs.edge_index)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        pooled = global_mean_pool(x, graphs.batch)
        projected = self.lins[1](pooled)

        output = projected.unsqueeze(1).expand(-1, self.config.intermediate_size, -1)

        return output
