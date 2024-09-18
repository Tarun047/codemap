from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from src.indexers.base import BaseIndexer
from typing import List
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from itertools import chain


class SourceCodeIndexer(BaseIndexer):
    def glob_pattern(self) -> str:
        return "**/*.cs"

    async def index_one(self, file_path: str) -> None:
        return super().index_one(file_path)

    async def index_few(self, file_paths: List[str]) -> None:
        loaders = [
            GenericLoader.from_filesystem(
                file_path,
                glob="**/*",
                parser=LanguageParser(language="csharp", parser_threshold=500),
            )
            for file_path in file_paths
        ]

        documents = [await loader.aload() for loader in loaders]
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.CSHARP, chunk_size=2000, chunk_overlap=200
        )
        snippets = code_splitter.split_documents(list(chain(*documents)))
        self.logger.info("Snippets:")
        for snippet in snippets:
            self.logger.info(snippet)
        await self.vector_db.aadd_documents(snippets)
