from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Node:
    id: str
    type: str
    metadata: Dict[str, str]
    relatives: Dict[str, List['Node']] = field(default_factory=dict)


class Graph:
    def __init__(self):
        self.node_map: Dict[str, Node] = dict()

    def add_node(self, id: str, type: str, metadata: Dict[str, str] = None):
        if metadata is None:
            metadata = dict()
        self.node_map[id] = Node(id, type, metadata)

    def add_edge(self, source_id, target_id, relationship_type: str):
        if source_id not in self.node_map:
            raise ValueError("Cannot find the source node in the graph")

        if target_id not in self.node_map:
            raise ValueError("Cannot find target node in the graph")

        source_node = self.node_map[source_id]
        if source_node.relatives is None:
            source_node.relatives = dict()

        if relationship_type not in source_node.relatives:
            source_node.relatives[relationship_type] = []

        source_node.relatives[relationship_type].append(self.node_map[target_id])
