from dataclasses import dataclass

from src.database.graph.configuration import DatabaseConfiguration
import json


@dataclass
class ApplicationConfiguration:
    graph_db: DatabaseConfiguration
    vector_db: DatabaseConfiguration
    nlp_model: str
    code_model: str
    code_embedding_model: str
    source_code_indexer_batch_size: int
    source_code_indexer_max_threads: int

    @staticmethod
    def get_instance() -> "ApplicationConfiguration":
        return ApplicationConfiguration(
            vector_db=DatabaseConfiguration(
                uri="localhost",
                port=8000,
                username='',
                database='',
                password=''
            ),
            graph_db=DatabaseConfiguration(
                uri='bolt://localhost:7687',
                username='neo4j',
                password='develop123',
                database='neo4j',
                port=0
            ),
            nlp_model='codellama',
            code_model='codellama',
            code_embedding_model='unclemusclez/jina-embeddings-v2-base-code',
            source_code_indexer_max_threads=16,
            source_code_indexer_batch_size=96
        )

