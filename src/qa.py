from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.schema import Document, StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import format_document
from langchain_core.runnables.base import RunnableSerializable


def get_docs(file_paths: list[str]) -> list[Document]:
    docs = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        pages = loader.load()  # PyPDFLoader returns a doc for each Page
        docs.append(Document(  # Let's merge all pages into a single text document
            page_content="\n".join(page.page_content for page in pages),
            metadata={"source": file_path}
        ))
    return docs


def build_qa_chain() -> RunnableSerializable:

    def build_context(docs: list[Document], doc_type: str="Document") -> str:
        doc_prompt = PromptTemplate.from_template("{page_content}")
        context = ""
        for i, doc in enumerate(docs):
            context += f"{doc_type} {i + 1}\n---\n{format_document(doc, doc_prompt)}\n\n"
        return context

    return (
        {
            "context": lambda input: build_context(input["docs"], input.get("doc_type", "Document")),
            "question": lambda input: input["question"]
        }  # Build the data to be injected into the prompt
        | PromptTemplate.from_template(
            "Use only the following context to answer the question at the end."
            "\nDo not use anything other than the context below to answer the question."
            "\nI'll repeat it is extremely important that you only use the provided context below to answer the question."
            "\nIf the context below is not sufficient to answer, just say that you don't know, don't try to make up an answer."
            "\n\nContext:\n\n{context}\n\nQuestion: {question}"
        )
        | ChatOllama(model="zephyr:7b-beta-q5_K_M")
        | StrOutputParser()
    )


def generate_answer(question: str, file_paths: list[str], **kwargs: dict) -> str:
    # Get Document objects with text content from file paths
    docs = get_docs(file_paths)

    # Construct QA chain
    qa_chain = build_qa_chain()

    # Prepare input
    input = {"docs":docs, "question":question}
    input.update(kwargs)

    # Invoke the chain
    return qa_chain.invoke(input)


if __name__ == "__main__":  # Entry point of the program
    print(generate_answer(
        question="Give me a short, 100 words summary of each candidate having a coursera certification in Generative AI. Please include their contact details as well.",
        file_paths=["~/Documents/Resumes/Pranav Dhoolia.pdf"],
        doc_type="Candidate Resume"
    ))