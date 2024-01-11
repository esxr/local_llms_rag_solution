import json

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.chat_models import ChatOllama


def extract_keywords(query: str) -> list[str]:

    # Construct the keyword extraction chain
    keyword_extraction_chain = (
        {
            "query": lambda query: query
        }
        | PromptTemplate.from_template(
            "From the query below please extract search keywords. "
            "The search keywords must be part of the query string. "
            "Avoid words general to this domain such as: candidates, resume, contact, and details. "
            "Output a json formatted list of keyword strings. "
            "Each item in the output json should be a string."
            "\n\nQuery:\n```{query}\n```"
        )
        | ChatOllama(model="zephyr:7b-beta-q5_K_M")
        | StrOutputParser()
    )

    # Invoke the chain
    response = keyword_extraction_chain.invoke(query)

    # Get the substring between last `[` and `]`
    return json.loads(f'[{response[response.rfind("[") + 1:response.rfind("]")].strip()}]')


if __name__ == "__main__":  # Entry point of the program
    print(extract_keywords("Give me a short (about 100 words) summary of candidates having coursera certification in Generative AI. Please include their contact details."))