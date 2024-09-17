from .configuration import GraphDatabaseConfiguration
from langchain_community.graphs.neo4j_graph import Neo4jGraph, GraphDocument
from langchain_community.graphs.graph_document import Node as N4JNode, Relationship as N4JRelationship
from langchain_core.documents.base import Document
from .models import Graph, Node


class GraphDatabase:
    def __init__(self, configuration: GraphDatabaseConfiguration):
        self.graph = Neo4jGraph(configuration.uri, username=configuration.username, password=configuration.password, database=configuration.database, enhanced_schema=True, refresh_schema=True, sanitize=True)

    def _create_n4j_node(self, node: Node) -> N4JNode:
        return N4JNode(id=node.id, type=node.type, metadata=node.metadata)

    def _create_n4j_relationship(self, source_node: N4JNode, target_node: N4JNode, type: str):
        return N4JRelationship(source=source_node, target=target_node, type=type)

    def _construct_graph_document(self, graph: Graph, source: str) -> GraphDocument:
        n4j_node_map = dict()
        nodes = []
        relationships = []
        source = Document(source)

        for source_id in graph.node_map:
            source_node = graph.node_map[source_id]
            if source_id not in n4j_node_map:
                n4j_node = self._create_n4j_node(source_node)
                n4j_node_map[source_id] = n4j_node
                nodes.append(n4j_node)

            # One level travel is okay here because at some point in the above for loop we will reach child node as well
            # But we still need to map parents relation to its immediate child
            for relation_type in source_node.relatives:
                relatives = source_node.relatives[relation_type]

                for relative in relatives:
                    if relative.id not in n4j_node_map:
                        n4j_node = self._create_n4j_node(relative)
                        n4j_node_map[relative.id] = n4j_node
                        nodes.append(n4j_node)

                    relation = self._create_n4j_relationship(n4j_node_map[source_id], n4j_node_map[relative.id], relation_type)
                    relationships.append(relation)

        document = GraphDocument(source=source, nodes=nodes, relationships=relationships)

        return document

    def save(self, source: str, graph: Graph):
        graph_document = self._construct_graph_document(graph, source)
        self.graph.add_graph_documents([graph_document], include_source=True)
