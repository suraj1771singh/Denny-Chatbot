from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from models import get_gemini_model, get_llama_model
from prompts import get_chat_prompt
from utils import bind_memory


def main():

    try:

        chat_model = get_llama_model()
        prompt = get_chat_prompt()
        parser = StrOutputParser()

        # Create the chain from the prompt and the model and parser
        chain = prompt | chat_model | parser

        # Add the memory to the chain
        store = {}
        config = {"configurable": {"session_id": "abc2"}}
        chain = bind_memory(store, chain)

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\nLeaving...")
                    break

                # Invoke the chain with the user's message
                res = chain.invoke(HumanMessage(user_input), config=config)
                print(f"Sarah: {res}")
            except KeyboardInterrupt:
                print("\nLeaving...")
                break
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
