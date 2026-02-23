import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import CohereRerank

load_dotenv(override=True)

index = os.getenv("PINECONE_INDEX_NAME")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
template = """
    You are a Senior ISO 27001 Auditor. Use only the provided context to answer accurately.
    If the context does not contain the answer, say 
    "I don’t have enough information in the provided documents." Do not add extra knowledge based outside the context.
    Answer the question directly based on the context.
    Keep it concise and professional.

    Context: {context}
    Question: {question}
    Answer:
    """
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs): #takes a list of documents and formats the page contents into a string, separated by blank spaces
    return "\n\n".join(d.page_content for d in docs)

def rerank_docs(inputs, *, reranker, top_n):
    docs = inputs["docs"]
    question = inputs["question"]

    if not docs:
        return {"docs": [], "question": question} #pass empty docs to prevent isses calling the reranker API

    top_docs = reranker.compress_documents(docs, question)[:top_n]
    return {"docs": top_docs, "question": question}

def get_compliance_chain():
    if not index:
        raise ValueError("Missing PINECONE_INDEX_NAME in environment variables.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(embedding=embeddings, index_name=index)

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    reranker = CohereRerank(model="rerank-english-v3.0")
    rerank_step = RunnableLambda(lambda x: rerank_docs(x, reranker=reranker, top_n=5))

    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    chain = (
        RunnableParallel(docs=base_retriever, question=RunnablePassthrough())
        | rerank_step
        | RunnableParallel(
            answer=answer_chain,
            sources=RunnableLambda(lambda x: x["docs"]),
        )
    )
    return chain  

if __name__ == "__main__":
    print("Retrieving...")
    chain = get_compliance_chain()
    print(chain.invoke("What information must be included in the Statement of Applicability according to ISO/IEC 27001:2022 clause 6.1.3?"))

