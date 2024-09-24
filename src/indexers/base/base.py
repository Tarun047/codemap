from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from typing import List
from src.indexers.utils import batched
from src.logging.base import BaseLoggerMixin
from langchain_chroma import Chroma


class BaseIndexer(ABC, BaseLoggerMixin):
    def __init__(
        self, repo_path: str, batch_size: int, max_threads: int, vector_db: Chroma
    ):
        super().__init__()
        self.repo_path = Path(repo_path).resolve()
        self.batch_size = batch_size
        self.max_threads = max_threads
        self.vector_db = vector_db

    @abstractmethod
    def glob_pattern(self) -> str:
        """
        rtype: str
        purpose: Returns the glob patterns relavant to this indexer.
        If a file path does not meet this glob pattern it won't be passed for your indexer.
        Example: **/*.c will recursively match all .c files.
        """
        raise NotImplementedError()

    @abstractmethod
    async def index_one(self, file_path: Path) -> None:
        """
        inputs:
            file_path: Takes the absolute path to a given file.
        rtype: None
        purpose: Defines how the indexer is going to index a single relavant file file. Adds the indexed file to the vector_db instance passed in the constructor.
        """
        raise NotImplementedError()

    async def index_few(self, file_paths: List[Path]) -> None:
        for file_path in file_paths:
            try:
                await self.index_one(file_path)
            except Exception as exc:
                self.logger.error("Inference failed", exc_info=exc)

    async def run(self) -> None:
        """
        purpose: Runs the indexing process using a multi process approach
        """
        if not self.repo_path.exists():
            # Well repo doesn't exist, so nothing to do, we bail out
            return

        glob_pattern = self.glob_pattern()
        candidate_documents = self.repo_path.glob(glob_pattern)

        with ThreadPoolExecutor(max_workers=self.max_threads, thread_name_prefix=threading.current_thread().name) as pool:
            pool.map(
                lambda batch: asyncio.run(self.index_few(batch)),
                batched(candidate_documents, self.batch_size),
            )








