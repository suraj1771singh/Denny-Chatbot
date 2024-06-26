from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embeddings():
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embedding

