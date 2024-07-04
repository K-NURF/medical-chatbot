from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

def get_embedding_function():
    embedding_function = OpenAIEmbeddings()
    # vector = embedding_function.embed_query("apple")
    # print(f"Vector for 'apple': {vector}")
    # print(f"Vector length: {len(vector)}")
    return embedding_function
    
