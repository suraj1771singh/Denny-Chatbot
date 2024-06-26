from langchain.chains.summarize import load_summarize_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.document_loaders import TextLoader
from langchain.chains.llm import LLMChain
from prompts import get_summary_prompt
from models import get_llama_model
from langchain_community.document_loaders import PyPDFDirectoryLoader


def main():
    try:
        model = get_llama_model()
        prompt = get_summary_prompt()
        parser = StrOutputParser()
        document_loader = TextLoader("content.txt")
        docs = document_loader.load()
        print("DOCS ARE HERE", docs)
        # chain = load_summarize_chain(modedl, chain_type="stuff")
        map_chain = LLMChain(llm=model, prompt=prompt)
        reduce_chain = LLMChain(llm=model, prompt=prompt)

        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )

        # Combines and iteratively reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=10000,
        )

        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
        )
        parser = StrOutputParser()

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=0
        )

        split_docs = text_splitter.split_documents(docs)
        print(map_reduce_chain.run(split_docs))
        # print(prompt)
        """   
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\nLeaving...")
                    break
                print(f"Sarah: ", end="")

            except KeyboardInterrupt:
                print("\nLeaving...")
                break
        """
    except Exception as e:
        print("ERROR : ", e)


if __name__ == "__main__":
    main()
