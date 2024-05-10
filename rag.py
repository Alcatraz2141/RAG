import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import OpenAIEmbeddings
import os
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import bs4
from dotenv import load_dotenv


load_dotenv()  # take environment variables from .env 
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="LangChain & Streamlit RAG")
st.title("LangChain & Streamlit RAG")

#process_url_clicked = st.sidebar.button("Process URLs")
#url = st.sidebar.text_input(f"URL")

#uploaded_file = st.file_uploader("Choose a file")
#if uploaded_file is not None:
#    bytes_data = uploaded_file.getvalue()
#    st.write("filename:", uploaded_file.name)
#    st.write(bytes_data)


def show_ui(qa, prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(prompt)
                st.markdown(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

parser = StrOutputParser()
system_prompt = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )


@st.cache_resource
def get_retriever(openai_api_key=OPENAI_API_KEY):
    loader=PyPDFLoader("2405.04517v1.pdf")
    docs=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','],chunk_size=1000,chunk_overlap=200)
    docus=text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectors=Chroma.from_documents(docus,embeddings)
    retriever = vectors.as_retriever()
    return retriever

def get_chain(openai_api_key=None, huggingfacehub_api_token=None):
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo')
    chain = ({"context": get_retriever(), "question": RunnablePassthrough()} | prompt| model | parser) 
    return chain

def ask_question( query):
    response = get_chain().invoke(
        query,
        config={"configurable": {"session_id": "foo"}}
    )
    return response



   
def run():
    ready=True
    if ready:
        chain = get_chain(openai_api_key=OPENAI_API_KEY)
        st.subheader("Ask me questions about anything")
        show_ui(chain, "What would you like to know?")
    else:
        st.stop()


run()