from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from logging import Logger
from math import ceil
from numpy import array_split
from pathlib import Path
import threading
from typing import List
from .utils import batched
from ..logging.base import BaseLoggerMixin

class BaseIndexer(ABC, BaseLoggerMixin):
    def __init__(self, 
            repo_path: str,
            batch_size: int, 
            max_threads: int
        ):
        super().__init__()
        self.repo_path = Path(repo_path).resolve()
        self.batch_size = batch_size
        self.max_threads = max_threads
    
    
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
    def index_one(self, file_path: Path) -> None:
        """
        inputs:
            file_path: Takes the absolute path to a given file.
        rtype: None
        purpose: Defines how the indexer is going to index a single relavant file file.
        """
        raise NotImplementedError()
    

    def index_few(self, file_paths: List[Path]) -> None:
        for file_path in file_paths:
            try:
                self.index_one(file_path)
            except Exception as exc:
                self.logger.error('Inference failed', exc_info=exc)

    def run(self) -> None:
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
                lambda batch: self.index_few(batch),
                batched(candidate_documents, self.batch_size),
            )








