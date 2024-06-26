# Python imports
from dotenv import load_dotenv
import logging

# Load API Keys
load_dotenv()

# Langchain imports

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from models import get_gemini_model
from tools import get_tools


# Set up logging
logging.getLogger().setLevel(logging.ERROR)

model = get_gemini_model()
tools = get_tools()

store = {}




agent_executor = create_react_agent(model, tools)
memory = SqliteSaver.from_conn_string(":memory:")





config = {"configurable": {"thread_id": "abc123"}}



chain = prompt | agent_executor
parser = StrOutputParser()

while True:
    try:
        user_input = input("You : ")
        print("AI :", end="")
        for r in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}, config
        ):

            print(r["agent"]["messages"][0].content, end="")
        print("")
    except KeyboardInterrupt:
        print("\nExiting...")
        # save_store()
        exit(0)
