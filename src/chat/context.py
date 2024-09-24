import chromadb
import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_community.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from src.chat.retrieval import PrintRetrievalHandler
from src.configuration.app import ApplicationConfiguration
from src.database.graph.core import GraphDatabase
from src.indexers.source.source_code_indexer import SourceCodeIndexer

st.set_page_config(page_title="Code Map")
st.title("Code Map: Speak to your repository :)")


class ChatContext:
    _instance: "ChatContext" = None

    def __init__(self):
        app_config = ApplicationConfiguration.get_instance()
        self.app_config = app_config
        self.nlp_slm = ChatOllama(model=app_config.nlp_model)
        self.code_slm = ChatOllama(model=app_config.code_model)
        self.indexed_repos = set()
        self.vector_db = Chroma(
            client=chromadb.HttpClient(
                host=app_config.vector_db.uri,
                port=app_config.vector_db.port
            ),
            embedding_function=OllamaEmbeddings(
                model=app_config.code_embedding_model
            )
        )
        self.graph_db = GraphDatabase(
            configuration=app_config.graph_db
        )
        self.history = StreamlitChatMessageHistory()
        self.doc_prompt = ChatPromptTemplate.from_messages(
            [
                ("placeholder", "{chat_history}"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
                ),
            ]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user's questions based on the below context:\n\n{context}",
                ),
                ("placeholder", "{chat_history}"),
                ("user", "{input}"),
            ]
        )

    @st.cache_resource(ttl="1h")
    async def index(_self, repo_path: str):
        indexers = [
            SourceCodeIndexer(
                repo_path=repo_path,
                batch_size=_self.app_config.source_code_indexer_batch_size,
                max_threads=_self.app_config.source_code_indexer_max_threads,
                vector_db=_self.vector_db
            )
        ]

        for indexer in indexers:
            await indexer.run()

    @property
    def retriever(self):
        return self.vector_db.as_retriever()

    @property
    def retriever_chain(self):
        return create_history_aware_retriever(self.code_slm, self.retriever, self.doc_prompt)

    @property
    def memory(self):
        return ConversationBufferMemory(memory_key="chat_history", chat_memory=self.history, return_messages=True)

    @property
    def document_chain(self):
        return create_stuff_documents_chain(self.code_slm, self.chat_prompt)

    @property
    def qa_chain(self):
        return create_retrieval_chain(self.retriever_chain, self.document_chain)

    def get_response(self, query):
        return self.qa_chain.stream({
            "chat_history": st.session_state.chat_history,
            "input": query
        })

    def stream_data(self, response):
        for chunk in response:
            if "answer" in chunk:
                yield chunk["answer"]

    async def run(self):
        """
        Entry point of the chat application
        :return:
        """
        repo_path = st.text_input(label="Please enter the local path to your codebase: ")
        if len(repo_path) == 0:
            return
        if repo_path not in self.indexed_repos:
            await self.index(repo_path)
            self.indexed_repos.add(repo_path)
        # session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, I am a bot. How can I help you?"),
            ]

        # conversation
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)

        # user input
        user_query = st.chat_input("Type your message here...")
        if user_query is not None and user_query != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            with st.chat_message("Human"):
                st.markdown(user_query)
            with st.chat_message("AI"):
                response = st.write_stream(self.stream_data(self.get_response(user_query)))
            st.session_state.chat_history.append(AIMessage(content=response))

    @staticmethod
    def get_instance():
        if ChatContext._instance is None:
            ChatContext._instance = ChatContext()
        return ChatContext._instance
