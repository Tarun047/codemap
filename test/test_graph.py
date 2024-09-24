from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

from src.database.graph.configuration import DatabaseConfiguration
from src.database.graph.core import GraphDatabase
from src.indexers.base import BaseIndexer
from src.indexers.dependency.external.python import PythonExternalDependencyIndexer
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from pathlib import Path

from src.indexers.dependency.internal.python import PythonInternalDependencyIndexer


class SampleIndexer(BaseIndexer):
    def glob_pattern(self) -> str:
        return "**/*.py"

    def index_one(self, file_path: Path) -> None:
        self.logger.info(file_path)
        # Extract all info
        # Filter whats needed - # /* */
        # Tokenize / Vectorize
        # Save it to vector DB / graph DB


config = DatabaseConfiguration('bolt://localhost:7687', username='neo4j', password='develop123', database='neo4j')
database = GraphDatabase(config)

indexer = PythonExternalDependencyIndexer(
    repo_path='../',
    graph_db=database,
    batch_size=10,
    max_threads=10
)

# indexer.run() NOTE: Uncomment this for indexing


CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:
# What are the dependencies of linky project?
MATCH (p:PYTHON_REQUIREMENTS_FILE {{id:'linky'}})-[:DEPENDS_ON_EXTERNAL]->(d:EXTERNAL_PACKAGE)
RETURN d.id as external_dependency

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

code_slm = ChatOllama(model="codellama")
chat_slm = ChatOllama(model="phi3")

chain = GraphCypherQAChain.from_llm(
    graph=database.graph,
    cypher_llm=code_slm,
    qa_llm=chat_slm,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT
)

print(chain.invoke({"query": "What are the dependencies of codemap project?"}))