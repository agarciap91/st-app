import pandas as pd
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.agents import Tool
from langchain.agents import create_json_chat_agent
from langchain import hub
from langchain.retrievers import ContextualCompressionRetriever
from libraries_initialization import PINECONE_API_KEY, INDEX_NAME, embeddings, llm

# Constants and configuration
def initialize_vector_store(index_name):
    """
    Initializes and returns a Pinecone Vector Store with specific index and embeddings.
    """
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    text_field = "text"
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index, embeddings, text_field)
    return vector_store

def retrieve_documents(query):
    """
    Retrieves documents relevant to a query using a vector store.
    """
    vector_store = initialize_vector_store(INDEX_NAME)
    retriever = vector_store.as_retriever(search_kwargs={'k': 20})
    documents = retriever.invoke(query)
    return documents

# Tool setup for LangChain
knowledge_base_tool = Tool(
    name='Knowledge Base',
    func=retrieve_documents,
    description='Usa esta base de datos vectorial para responder a preguntas relacionadas con medicina y principios de cirugia'
)

# Chat agent configuration and initialization
tools = [knowledge_base_tool]
chat_prompt = hub.pull("hwchase17/react-chat-json")
chat_agent = create_json_chat_agent(llm=llm, tools=tools, prompt=chat_prompt)