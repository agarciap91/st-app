
import boto3
import json
import os
#from langchain.memory import DynamoDBChatMessageHistory
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
#from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrock
import streamlit as st


# Constants for configuration
REGION_NAME = 'us-east-1'
REGION_NAME_BEDROCK = 'us-east-1'
INDEX_NAME = 'test-ai-pdf'
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# Setup AWS boto3 session and clients
aws_session = boto3.Session(region_name=REGION_NAME,    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"])
#aws_session = boto3.Session(region_name=REGION_NAME)
secrets_manager_client = aws_session.client(service_name='secretsmanager')
bedrock_client = boto3.client("bedrock-runtime", region_name=REGION_NAME_BEDROCK)

# Initialize Langchain components
embeddings = BedrockEmbeddings(client=bedrock_client, region_name=REGION_NAME_BEDROCK,model_id="amazon.titan-embed-text-v2:0" )
llm = ChatBedrock(model_id=MODEL_ID, region_name=REGION_NAME_BEDROCK, client=bedrock_client)


PINECONE_API_KEY = aws_access_key_id=st.secrets["PINECONE_API_KEY"]  

