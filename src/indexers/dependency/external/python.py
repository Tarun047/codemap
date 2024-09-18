from src.indexers.base import BaseIndexer
from pathlib import Path
from src.database.graph.models import Graph
from src.database.graph.core import GraphDatabase
from os import sep
import requirements


class PythonExternalDependencyIndexer(BaseIndexer):
    def __init__(self, repo_path: str, graph_db: GraphDatabase, batch_size: int, max_threads: int):
        super().__init__(repo_path, batch_size, max_threads)
        self.graph_db = graph_db

    def glob_pattern(self) -> str:
        return '**/requirements.txt'

    def index_one(self, file_path: Path) -> None:
        graph = Graph()
        resolved_path = file_path.resolve()
        source_package_id = resolved_path.parent.name
        package_name = resolved_path.parent.name
        metadata = dict()
        metadata['package_name'] = package_name
        metadata['location'] = resolved_path

        graph.add_node(source_package_id, 'PYTHON_REQUIREMENTS_FILE', metadata)
        with resolved_path.open() as infile:
            dependencies = requirements.parse(infile)
            for dependency in dependencies:
                dependent_package_id = dependency.name
                metadata = dict()
                metadata['name'] = dependency.name
                metadata['specs'] = str(dependency.specs)
                metadata['uri'] = str(dependency.uri)
                metadata['vcs'] = dependency.vcs

                graph.add_node(dependent_package_id, 'EXTERNAL_PACKAGE', metadata)
                graph.add_edge(source_package_id, dependent_package_id, 'DEPENDS_ON_EXTERNAL')

        self.graph_db.save(str(file_path.absolute()), graph)
