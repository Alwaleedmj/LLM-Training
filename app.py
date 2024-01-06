import streamlit as st
import io
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemp import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_text(uploaded_file):
    # Read text from the uploaded file-like object
    # Using io.StringIO to handle the text data
    text_io = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    return text_io.read()

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title='My Streamlit App', page_icon=':shark:', layout='wide')
    st.write(css, unsafe_allow_html=True)


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat with txt')
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)


    with st.sidebar:
        st.subheader('Document')
        uploaded_files = st.file_uploader("Choose a file", key="file_uploader1", accept_multiple_files=True)
        if st.button('Upload'):
            with st.spinner('Uploading...'):
                for uploaded_file in uploaded_files:
                    #Get the raw text
                    raw_text = get_text(uploaded_file)
                    # Get the chunk
                    text_chunks = get_text_chunks(raw_text)
                    # get the vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # Start conversation 
                    st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
