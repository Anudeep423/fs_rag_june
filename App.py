import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate  
from langchain.chains import RetrievalQA

st.title("Full Stack Web Chatbot")
st.sidebar.header("Please enter secret info")
api_key = st.sidebar.text_input("OpenAI_API_Key", type="password")
st.header("Please ask any question related to full stack website below")
question = st.text_input("Enter your question here:")

@st.cache_resource(show_spinner="Loading and processing website...")
def load_vector_db():
    urls = ["https://fullstackacademy.in"]
    loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    chunks = text_splitter.split_documents(data)
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings_model)
    return vector_db

if api_key:
    try:
        vector_database = load_vector_db()
        llm = OpenAI(api_key=api_key)

        template = """Use the context strictly to provide a concise answer. If you don't know, just say you don't know.
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_database.as_retriever(),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        if question:
            with st.spinner("Generating answer..."):
                answer = chain.run(question)
            st.subheader("Generated Answer:")
            st.write(answer)

    except Exception as e:
        st.error(f"Error occurred during execution: {e}")
else:
    st.warning("Please enter your OpenAI API Key")
