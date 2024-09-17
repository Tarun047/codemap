from src.indexers.base import BaseIndexer

class SampleIndexer(BaseIndexer):
    def glob_pattern(self) -> str:
        return "**/*.py"
    
    def index_one(self, file_path: str) -> None:
        self.logger.info(file_path)


indexer = SampleIndexer(
    repo_path='../',
    batch_size=10,
    max_threads=10,
)

indexer.run()