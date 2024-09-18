import asyncio
from src.indexers.source_code_indexer import SourceCodeIndexer
import chromadb
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

REPO_PATH = "Q:\src\Substrate\sources\dev\Hygiene\src\Common\Antispam\Core\Optics\TypePartitioned"

async def main():
    # create vector db
    client = chromadb.HttpClient("localhost", port=8000)
    vector_db = Chroma(
        client=client,
        embedding_function=OllamaEmbeddings(
            model="unclemusclez/jina-embeddings-v2-base-code"
        ),
    )
    # index the source code and add it to vector db
    indexer = SourceCodeIndexer(
        repo_path=REPO_PATH, batch_size=10, max_threads=10, vector_db=vector_db
    )

    await indexer.run()

    # retrieval
    retriever = vector_db.as_retriever()

    # augment and generate
    llm = ChatOllama(model="codellama")

    # First we need a prompt that we can pass into an LLM to generate this search query

    prompt = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)

    qa = create_retrieval_chain(retriever_chain, document_chain)

    for i in range(10):
        question = input("Enter prompt:")
        result = qa.invoke({"input": question})
        print(result["answer"])

if __name__ == "__main__":
    asyncio.run(main())
