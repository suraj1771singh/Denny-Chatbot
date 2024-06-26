from dotenv import load_dotenv

# Load API Keys
load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from models import get_llama_model, get_gemini_model
from tools import get_tools
from prompts import get_chat_prompt


def main():
    try:
        chat_model = get_llama_model()
        tools = get_tools()
        prompt = hub.pull("hwchase17/react")
        parser = StrOutputParser()
        agent = create_react_agent(model=chat_model, tools=tools)
        # model_with_tools = chat_model.bind_tools(tools)
        # response = agent.invoke(
        #     {"messages": [HumanMessage(content="What is the capital of France?")]}
        # )
        # print(response["messages"])

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\nLeaving...")
                    break
                res = agent.invoke({"messages": [HumanMessage(content=user_input)]})
                # content = parser.invoke(res["messages"][-1])
                print(res)
                print("OUTPUT :", res["messages"][-1].content)
                # print(f"Sarah: {res}")
            except KeyboardInterrupt:
                print("\nLeaving...")
                break

    except Exception as e:
        print("ERROR : ", e)


if __name__ == "__main__":
    main()
