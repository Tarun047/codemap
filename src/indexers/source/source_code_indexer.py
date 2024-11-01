from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from src.indexers.base import BaseIndexer
from typing import List
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from itertools import chain
from pathlib import Path


class SourceCodeIndexer(BaseIndexer):
    def glob_pattern(self) -> str:
        return "**/*.py"

    async def index_one(self, file_path: Path) -> None:
        return super().index_one(file_path)

    async def index_few(self, file_paths: List[Path]) -> None:
        loaders = [
            GenericLoader.from_filesystem(
                file_path,
                glob="**/*",
                parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
                show_progress=True,
            )
            for file_path in file_paths
        ]

        documents = [await loader.aload() for loader in loaders]
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
        )
        snippets = code_splitter.split_documents(list(chain(*documents)))
        self.logger.info(f"Indexing documents: ${file_paths}")
        await self.vector_db.aadd_documents(snippets)
