from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
import bs4
from langchain_community.document_loaders import WebBaseLoader

# Load HTML
loader = AsyncHtmlLoader(["https://en.wikipedia.org/wiki/India"])
html = loader.load()
# Transform
print(html)
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(
    html, tags_to_extract=["p", "div"]
)
print(
    "-------------------------------------------------------------------------------------"
)
# print(docs_transformed[0].page_content[:])
print(
    "-------------------------------------------------------------------------------------"
)
# bs4_strainer = bs4.SoupStrainer(["p", "div"])
# loader = WebBaseLoader(
#     web_paths=("https://timesofindia.indiatimes.com/india/timestopten.cms",),
#     bs_kwargs={"parse_only": bs4_strainer},
# )
# docs = loader.load()
# print(docs[0].page_content[:])
