import os
import streamlit as st
import time
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

store_dir = "faiss_store_openai"
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

run_url_clicked = st.sidebar.button("Run")

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=400)
embeddings = OpenAIEmbeddings()

if run_url_clicked:
    # Load data from the provided URLs.
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split the data into manageable chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create the FAISS vector store from documents.
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")

    # Save the vector store using FAISS's native persistence.
    vectorstore_openai.save_local(store_dir)
    main_placeholder.text("FAISS vector store saved locally.")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(store_dir):
        # Load the FAISS vector store with dangerous deserialization enabled.
        vectorstore = FAISS.load_local(store_dir, embeddings, allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever()
        )
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
