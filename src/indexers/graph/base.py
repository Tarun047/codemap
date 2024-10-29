from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_experimental.graph_transformers.llm import UnstructuredRelation

from src.database.graph.core import GraphDatabase
from src.indexers.base import BaseIndexer


def create_unstructured_prompt(
        node_labels: Optional[List[str]] = None, rel_types: Optional[List[str]] = None
) -> ChatPromptTemplate:
    node_labels_str = str(node_labels) if node_labels else ""
    rel_types_str = str(rel_types) if rel_types else ""
    base_string_parts = [
        "You are a top-tier algorithm designed for extracting information in "
        "structured formats to build a knowledge graph. Your task is to identify "
        "the entities and relations requested with the user prompt from a source code."
        "You must generate the output in a JSON format containing a list "
        'with JSON objects. Each object should have the keys: "head", '
        '"head_type", "relation", "tail", and "tail_type". The "head" '
        "key must contain the text of the extracted entity with one of the types "
        "from the provided list in the user prompt.",
        f'The "head_type" key must contain the type of the extracted head entity, '
        f"which must be one of the types from {node_labels_str}."
        if node_labels
        else "",
        f'The "relation" key must contain the type of relation between the "head" '
        f'and the "tail", which must be one of the relations from {rel_types_str}.'
        if rel_types
        else "",
        f'The "tail" key must represent the text of an extracted entity which is '
        f'the tail of the relation, and the "tail_type" key must contain the type '
        f"of the tail entity from {node_labels_str}."
        if node_labels
        else "",
        "Attempt to extract as many entities and relations as you can."
        "When extracting entities and forming relationships from source code you should consider to capture"
        "variable names and their usage, class inheritance hierarchy, method call hierarchy, method return values, package import and export usage patterns, design patterns used."
        "IMPORTANT NOTES:\n- Don't add any explanation and text.",
    ]
    system_prompt = "\n".join(filter(None, base_string_parts))

    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=UnstructuredRelation)

    human_string_parts = [
        "Based on the following example, extract entities and "
        "relations from the provided text.\n\n",
        "Use the following entity types, don't use other entity "
        "that is not defined below:"
        "# ENTITY TYPES:"
        "{node_labels}"
        if node_labels
        else "",
        "Use the following relation types, don't use other relation "
        "that is not defined below:"
        "# RELATION TYPES:"
        "{rel_types}"
        if rel_types
        else "",
        "For the following source code, extract entities and relations"
        "{format_instructions}\nText: {input}",
    ]
    human_prompt_string = "\n".join(filter(None, human_string_parts))
    human_prompt = PromptTemplate(
        template=human_prompt_string,
        input_variables=["input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "node_labels": node_labels,
            "rel_types": rel_types,
        },
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message, human_message_prompt]
    )
    return chat_prompt


class GraphIndexer(BaseIndexer):
    def glob_pattern(self) -> str:
        return "**/*.py"

    async def index_one(self, file_path: Path) -> None:
        pass

    def __init__(self, repo_path: str, batch_size: int, max_threads: int, vector_db: Chroma, graph_db: GraphDatabase,
                 slm: BaseLanguageModel):
        super().__init__(repo_path, batch_size, max_threads, vector_db)
        self.graph_db = graph_db
        parser = JsonOutputParser(pydantic_object=UnstructuredRelation)
        self.prompt = create_unstructured_prompt()
        self.graph_transformer = LLMGraphTransformer(llm=slm, prompt=self.prompt)

    async def index_few(self, file_paths: List[Path]) -> None:
        try:
            self.logger.info(f"Graph:: Indexing paths: {file_paths}")
            documents = [Document(page_content=file_path.read_text()) for file_path in file_paths]
            graph_docs = self.graph_transformer.convert_to_graph_documents(documents)
            self.graph_db.save_all(graph_docs)
        except Exception as e:
            self.logger.error(e)
