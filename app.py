import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma

# --- 1. UI SETUP ---
st.set_page_config(page_title="Free RAG Document Intelligence", layout="wide")
st.title("📄 Free RAG Document Intelligence (100% Local)")

# Sidebar for PDF Upload
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Extracting, Chunking, and Embedding locally..."):
            # Save file temporarily
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # --- 2. EXTRACT TEXT ---
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            # --- 3. CHUNK DOCUMENTS ---
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            
            # --- 4 & 5. FREE EMBEDDINGS & STORE IN VECTOR DB ---
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Clear old data if exists
            if "vectorstore" in st.session_state:
                st.session_state.vectorstore.delete_collection()
                
            vectorstore = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings,
                collection_name="free_doc_intelligence"
            )
            
            st.session_state.vectorstore = vectorstore
            st.success(f"Success! Processed {len(chunks)} chunks locally.")
            os.remove(temp_path)

# --- MAIN CHAT INTERFACE ---
if "vectorstore" not in st.session_state:
    st.info("👈 Please upload a PDF and click 'Process Document' to begin.")
else:
    st.header("2. Ask Questions")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking locally..."):
                # --- 6. RETRIEVE RELEVANT CHUNKS ---
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="similarity", 
                    search_kwargs={"k": 4}
                )
                docs = retriever.invoke(prompt)
                
                # --- 7. FORMAT CONTEXT MANUALLY (No chains needed) ---
                context = ""
                for i, doc in enumerate(docs):
                    page_num = doc.metadata.get("page", "Unknown")
                    context += f"Source {i+1} (Page {page_num}):\n{doc.page_content}\n\n"

                # --- 8. SEND TO LLM ---
                llm = ChatOllama(model="llama3.2", temperature=0)
                
                system_prompt = (
                    "You are an expert document analyst. Use the following pieces of retrieved context "
                    "to answer the question. If you don't know the answer, say that you don't know. "
                    "Use clear, concise language and cite the page numbers from the context (e.g., [Page 3]).\n\n"
                    f"CONTEXT:\n{context}"
                )
                
                messages = [
                    ("system", system_prompt),
                    ("human", prompt)
                ]
                
                response = llm.invoke(messages)
                answer = response.content
                
                st.markdown(answer)
                
                # --- 9. SHOW CITATIONS ---
                if docs:
                    with st.expander("📚 View Sources & Citations"):
                        for i, doc in enumerate(docs):
                            page_num = doc.metadata.get("page", "Unknown")
                            st.markdown(f"**Source {i+1} (Page {page_num}):**")
                            st.text(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})