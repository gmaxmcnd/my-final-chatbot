# app.py (최종 완성 버전)
import streamlit as st
from google.oauth2 import service_account
from google.cloud import firestore
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- Firestore 인증 정보 설정 ---
# 이 부분은 Streamlit Cloud의 Secrets 설정을 통해 안전하게 처리됩니다.
try:
    creds_dict = {
        "type": st.secrets.firestore.type,
        "project_id": st.secrets.firestore.project_id,
        "private_key_id": st.secrets.firestore.private_key_id,
        "private_key": st.secrets.firestore.private_key.replace('\\n', '\n'),
        "client_email": st.secrets.firestore.client_email,
        "client_id": st.secrets.firestore.client_id,
        "auth_uri": st.secrets.firestore.auth_uri,
        "token_uri": st.secrets.firestore.token_uri,
        "auth_provider_x509_cert_url": st.secrets.firestore.auth_provider_x509_cert_url,
        "client_x509_cert_url": st.secrets.firestore.client_x509_cert_url
    }
    creds = service_account.Credentials.from_service_account_info(creds_dict)
    db = firestore.Client(credentials=creds)

except Exception as e:
    st.error("⚠️ Firestore 인증 정보 설정에 문제가 있습니다. Streamlit Cloud의 Secrets 설정을 확인해주세요.")
    st.error(f"오류 상세: {e}")
    st.stop()

# --- 이하 챗봇 로직 (이전과 동일) ---
def load_documents_from_firestore():
    docs_ref = db.collection('documents').stream()
    documents_list = [Document(page_content=doc.to_dict().get('content', ''), metadata={'source': doc.to_dict().get('source', '')}) for doc in docs_ref]
    return documents_list

@st.cache_resource(ttl="10m")
def setup_rag_pipeline_from_db():
    with st.spinner("지식 데이터베이스를 로딩하고 있습니다..."):
        docs = load_documents_from_firestore()

    if not docs:
        st.error("Firestore에서 문서를 불러오지 못했습니다. DB에 데이터가 있는지 확인해주세요.")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    vector_db = FAISS.from_documents(documents=split_documents, embedding=embedding_model)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=st.secrets["GOOGLE_API_KEY"])

    prompt = ChatPromptTemplate.from_template("Context:\n{context}\n\nQuestion: {input}\n\nAnswer:")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain, datetime.now()

# --- Streamlit UI 설정 ---
st.title("📜 제미니 AI 연구소 규정 안내 챗봇")
st.markdown("---")

try:
    retrieval_chain, cached_time = setup_rag_pipeline_from_db()
    st.success(f"지식 DB가 {cached_time.strftime('%H시 %M분 %S초')}에 업데이트 되었습니다.")

    user_query = st.text_input("질문을 입력하세요:", key="user_query")

    if user_query:
        with st.spinner("Gemini가 답변을 생성하는 중입니다..."):
            response = retrieval_chain.invoke({"input": user_query})
            st.markdown("#### 답변:")
            st.write(response["answer"])

except Exception as e:
    st.error(f"답변 생성 중 오류가 발생했습니다: {e}")