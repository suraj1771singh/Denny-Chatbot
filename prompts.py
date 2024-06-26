from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub


def get_chat_prompt():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("ai", "You are personal assistant and your name is Sarah!"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt


def get_rag_prompt():
    prompt = hub.pull("rlm/rag-prompt")
    return prompt


def get_summary_prompt():
    prompt = hub.pull("rlm/map-prompt")
    return prompt
