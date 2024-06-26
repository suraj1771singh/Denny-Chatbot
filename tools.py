from langchain_community.tools.tavily_search import TavilySearchResults

list_of_tools = []

# Search Tool
search = TavilySearchResults(max_results=2)
list_of_tools.append(search)


def get_tools():
    return list_of_tools
