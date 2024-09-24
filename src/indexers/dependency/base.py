import abc
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

from src.configuration.app import ApplicationConfiguration
from src.database.graph.core import GraphDatabase
from src.indexers.base import BaseIndexer


class BaseDependencyParser(BaseIndexer, abc.ABC):
    def __init__(self, repo_path: str, batch_size: int, max_threads: int, vector_db: Chroma, graph_db: GraphDatabase):
        super().__init__(repo_path, batch_size, max_threads, vector_db)
        self.graph_db = graph_db
        app_cfg = ApplicationConfiguration.get_instance()
        self.graph_transformer = LLMGraphTransformer(llm=ChatOllama(app_cfg.code_model))

    async def index_few(self, file_paths: List[Path]) -> None:
        documents = [Document(page_content=path.read_text()) for path in file_paths]
        graph_docs = await self.graph_transformer.aconvert_to_graph_documents(documents)
        self.graph_db.save_all(graph_docs)
