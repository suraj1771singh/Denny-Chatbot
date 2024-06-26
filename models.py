from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint


def get_gemini_model():
    return ChatGoogleGenerativeAI(model="gemini-pro")


def get_llama_model():
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = HuggingFaceEndpoint(repo_id=repo_id)
    return ChatHuggingFace(llm=llm)
