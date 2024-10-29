from pathlib import Path

from langchain_chroma import Chroma

from src.database.graph.core import GraphDatabase
from src.indexers.base import BaseIndexer

class CsharpDependencyIndexer(BaseIndexer):
    def __init__(self, repo_path: str, batch_size: int, max_threads: int, vector_db: Chroma, graph_db: GraphDatabase):
        super().__init__(repo_path, batch_size, max_threads, vector_db)

    async def index_one(self, file_path: Path) -> None:
        pass

    def glob_pattern(self) -> str:
        pass