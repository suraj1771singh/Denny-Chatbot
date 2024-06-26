from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


def bind_memory(store: dict, chain):
    def get_session_history(session_id):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    with_message_history = RunnableWithMessageHistory(chain, get_session_history)
    return with_message_history
