import streamlit as st
from langchain_aws import ChatBedrock
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="NZNUZ5NRQ6",
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 10}},
)

prompt = ChatPromptTemplate.from_template(
    "以下のcontextに基づいて回答してください： {context} / 質問 {question}"
)

model = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"max_tokens": 1000},
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

st.title("おしえて！Bedrock")
question = st.text_input("質問を入力")
button = st.button("質問する")

if button:
    st.write(chain.invoke(question))