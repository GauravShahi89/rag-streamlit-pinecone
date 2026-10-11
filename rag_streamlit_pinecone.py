from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

INDEX_NAME= "qa-knowledge-base-1"


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

pc =Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index=pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k":3})

st.title("Quick Resolution to your Query")
query = st.text_input("What is your Query: ", placeholder="Enter your query here..")

llm=ChatOpenAI(model="gpt-5.4",temperature=0)
if st.button("Generate Response") and query:
    content=[]
    Question_Answer_Chunk = retriever.invoke(query)
    for doc in Question_Answer_Chunk:
        content.append(doc.page_content)

    context = "\n".join(content)

    prompt = f"""
            Based on user {query},provide the output only from the received {context}
            Rules:
            1. Do not use information outside the context.
            2. If the exact item asked by the user is not mentioned,
                check whether the context provides a list of supported options.
            3. If the requested option is not in that list, clearly say that
                it is not listed as supported and mention the supported options.
            4. If the context truly does not contain enough information,
                say "Information not available in the knowledge base."
            """

    with st.spinner("Generating Response"):
        response = llm.invoke(prompt)
        st.write(response.content)
