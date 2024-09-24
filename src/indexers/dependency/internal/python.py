import re
from pathlib import Path

from langchain_chroma import Chroma

from src.database.graph.core import GraphDatabase
from src.database.graph.models import Graph
from src.indexers.base import BaseIndexer
import list_imports


class PythonInternalDependencyIndexer(BaseIndexer):
    def __init__(self, repo_path: str, batch_size: int, max_threads: int, vector_db: Chroma, graph_db: GraphDatabase):
        super().__init__(repo_path, batch_size, max_threads, vector_db)
        self.graph_db = graph_db
        self.ignore_patterns = (
            re.compile(r'.*venv.*'),
            re.compile(r'.*site-packages.*'),
            re.compile(r'.*node_modules.*')
        )

    def glob_pattern(self) -> str:
        return '**/*.py'

    def index_one(self, file_path: Path) -> None:
        resolved_path = file_path.resolve()
        resolved_path_str = str(file_path.resolve())
        for pattern in self.ignore_patterns:
            if pattern.match(resolved_path_str):
                return

        self.logger.info(f"Resolving internal python dependencies for {file_path}")
        graph = Graph()

        source_file_id = str(resolved_path.relative_to(self.repo_path))
        metadata = dict()
        metadata['python_file_name'] = resolved_path.name
        metadata['location'] = resolved_path

        graph.add_node(source_file_id, 'PYTHON_SOURCE_FILE', metadata)
        dependencies = list_imports.get(str(resolved_path))
        for dependency in dependencies:
            dependent_package_id = dependency
            metadata = dict()
            metadata['name'] = dependency

            graph.add_node(dependent_package_id, 'IMPORT_REFERENCE', metadata)
            graph.add_edge(source_file_id, dependent_package_id, 'DEPENDS_ON_INTERNAL')

        self.graph_db.save(str(file_path.absolute()), graph)
