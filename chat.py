from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from models import get_gemini_model, get_llama_model
from prompts import get_chat_prompt
from utils import bind_memory
from typing import Dict, Any


class Chat:
    def __init__(self) -> None:
        self.chat_model = get_llama_model()
        self.prompt = get_chat_prompt()
        self.parser = StrOutputParser()

        # Create the chain from the prompt and the model and parser
        self.chain = self.prompt | self.chat_model | self.parser

        # Add the memory to the chain
        self.store: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {"configurable": {"session_id": "abc2"}}
        self.chain = bind_memory(self.store, self.chain)

    def generate_response(self, user_input: str):
        try:
            # Invoke the chain with the user input and configuration
            response = self.chain.stream(HumanMessage(user_input), config=self.config)
            return response
        except Exception as e:
            # Handle any exceptions gracefully
            return f"Error occurred: {str(e)}"


# def main():

#     try:

#         chat_model = get_llama_model()
#         prompt = get_chat_prompt()
#         parser = StrOutputParser()

# while True:
#     try:
#         user_input = input("You: ")
#         if user_input.lower() in ["exit", "quit", "bye"]:
#             print("\nLeaving...")
#             break

#         # Invoke the chain with the user's message
#         res = chain.invoke(HumanMessage(user_input), config=config)
#         print(f"Sarah: {res}")
#     except KeyboardInterrupt:
#         print("\nLeaving...")
#         break
#     except Exception as e:
#         print(e)


# if __name__ == "__main__":
#     main()
