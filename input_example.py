from typing import List, Union

from sentence_transformers import InputExample


class InputExampleWithGraph(InputExample):
    """
    Extended InputExample that includes graph-related data (edge_index, edge_type, pos_ids)
    """

    def __init__(
        self,
        guid: str = "",
        texts: List[str] = None,
        label: Union[int, float] = 0,
        edge_index=None,
        edge_type=None,
        pos_ids=None,
    ):
        """
        Creates one InputExample with the given texts, guid, label, and additional graph-related information.

        :param guid: id for the example
        :param texts: the texts for the example.
        :param label: the label for the example
        :param edge_index: edge indices for a graph
        :param edge_type: edge types for a graph
        :param pos_ids: position ids for the graph
        """
        super().__init__(guid=guid, texts=texts, label=label)

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.pos_ids = pos_ids

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str}, edge_index: {self.edge_index}, edge_type: {self.edge_type}, pos_ids: {self.pos_ids}"

    def set_label(self, label):
        self.label = label
