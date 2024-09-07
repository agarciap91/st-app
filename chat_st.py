import streamlit as st
import time
import pandas as pd
from uuid import uuid4
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from chat_retrieval import chat_agent, tools


# Execute chat agent
def execute_chat_agent(user_input, memory):
    """
    Execute the chat agent with the given input and memory.
    
    Args:
    user_input (str): User's input message
    memory (ConversationBufferMemory): Chat memory object

    Returns:
    dict: Chat agent's response
    """
    agent_executor = AgentExecutor(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    try:
        return agent_executor.invoke({"input": user_input}, {"callbacks": [st_cb]})
    except Exception as e:
        st.error(f"An error occurred while executing the chat agent: {e}")
        return None

# Display chat response
def display_chat_response(response, message_history):
    """
    Display the chat agent's response and sources.
    
    Args:
    response (dict): Chat agent's response
    message_history (StreamlitChatMessageHistory): Chat message history
    """
    with st.chat_message("assistant"):
        st.write(response["output"])


# Main chat interface
def run_chat_interface():
    """
    Run the main chat interface using Streamlit.
    """
    st.header("Cirugia General Assistant App")

    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=msgs,
        return_messages=True,
        memory_key="chat_history",
        output_key="output"
    )

    if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
        msgs.clear()
        st.session_state['steps'] = {}

    # Display chat history
    avatars = {"human": "user", "ai": "assistant"}
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type]):
            st.write(msg.content)

    # Handle user input
    if prompt := st.chat_input(placeholder="What is the Education Federal Student Aid?"):
        st.chat_message("user").write(prompt)        
        response = execute_chat_agent(prompt, memory)
        
        if response:
            display_chat_response(response, msgs)
            

# Main function
def main():
    """
    Main function to run the Streamlit app.
    """
    try:
        run_chat_interface()
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()