import asyncio

import chromadb
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

REPO_PATH = "CsharpRepo"


async def main():
    loader = GenericLoader.from_filesystem(
        REPO_PATH,
        glob="**/*",
        suffixes=[".cs", ".csproj"],
        parser=LanguageParser(language="csharp", parser_threshold=500)
    )

    documents = await loader.aload()

    code_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.CSHARP,
        chunk_size=2000,
        chunk_overlap=200
    )
    snippets = code_splitter.split_documents(documents)

    client = chromadb.HttpClient("localhost", port=8000)
    chroma = Chroma(client=client,
                    embedding_function=OllamaEmbeddings(model='unclemusclez/jina-embeddings-v2-base-code'))
    await chroma.aadd_documents(snippets)

    retriever = chroma.as_retriever()

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

    question = "What is the Main class in Semphie namespace trying to do?"
    result = qa.invoke({"input": question})
    print(result["answer"])


if __name__ == '__main__':
    asyncio.run(main())
