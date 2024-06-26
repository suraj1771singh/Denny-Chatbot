from dotenv import load_dotenv

load_dotenv()

from embeddings import get_embeddings
from langchain_community.vectorstores import Chroma
from models import get_llama_model
from prompts import get_rag_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_PATH = "db"


def main():
    try:
        embedding = get_embeddings()
        model = get_llama_model()
        prompt = get_rag_prompt()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\nLeaving...")
                    break

                # Invoke the chain with the user's message
                print(f"Sarah: ", end="")
                for chunk in rag_chain.stream(user_input):
                    print(chunk, end="", flush=True)
                print()

            except KeyboardInterrupt:
                print("\nLeaving...")
                break
    except Exception as e:
        print(e)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


if __name__ == "__main__":
    main()
