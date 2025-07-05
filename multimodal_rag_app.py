# app.py 또는 multimodal_rag_app.py 파일의 맨 위에 추가하세요.
import streamlit as st

st.set_page_config(layout="wide")
st.title("🐞 Streamlit Secrets 디버깅")
st.header("현재 앱이 인식하고 있는 Secrets 목록:")

# st.secrets의 모든 내용을 화면에 그대로 출력합니다.
# 이렇게 하면 어떤 이름으로 저장되었는지 정확히 알 수 있습니다.
st.write(st.secrets.to_dict())

st.header("분석:")
if "firestore" in st.secrets:
    st.success("✅ [firestore] 섹션을 성공적으로 찾았습니다! 이제 나머지 코드가 실행되어야 합니다.")
else:
    st.error("❌ [firestore] 섹션을 찾지 못했습니다. Streamlit Cloud 대시보드의 Secrets 설정을 다시 확인해주세요. 섹션 이름에 오타가 없는지, 대소문자가 올바른지(전부 소문자), 불필요한 공백이 없는지 확인이 필요합니다.")

# --- (기존의 나머지 앱 코드는 이 아래에 그대로 둡니다) ---
# 예: from google.oauth2 import service_account ... 등

# multimodal_rag_app.py (프로젝트 지정 오류 해결된 최종 버전)
import streamlit as st
from google.oauth2 import service_account
from google.cloud import firestore
from datetime import datetime
import base64
import mimetypes
import vertexai # <--- import 추가

from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from PIL import Image

# --- 인증 및 초기화 ---
try:
    # Streamlit Secrets에서 각 항목을 개별적으로 읽어와 딕셔너리로 조립
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
    
    # Firestore 클라이언트 초기화
    db = firestore.Client(credentials=creds, project=st.secrets.firestore.project_id)

    # Vertex AI 전역 초기화 (가장 중요!)
    vertexai.init(project=st.secrets.firestore.project_id, credentials=creds)

except Exception as e:
    st.error("⚠️ 인증 정보 설정에 문제가 있습니다. Streamlit Cloud의 Secrets 설정을 확인해주세요.")
    st.error(f"오류 상세: {e}")
    st.stop()

# --- 헬퍼 함수 및 핵심 로직 (이하 부분은 이전과 동일) ---
def image_to_base64_uri(file_path_or_bytes):
    if isinstance(file_path_or_bytes, str):
        mime_type, _ = mimetypes.guess_type(file_path_or_bytes)
        if mime_type is None: mime_type = 'image/jpeg'
        with open(file_path_or_bytes, "rb") as image_file:
            bytes_data = image_file.read()
    else:
        mime_type = file_path_or_bytes.type
        bytes_data = file_path_or_bytes.getvalue()
    encoded_string = base64.b64encode(bytes_data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"

@st.cache_resource
def load_retriever():
    embedding_model = VertexAIEmbeddings(model_name="multimodalembedding@001")
    vector_store = FAISS.load_local("faiss_multimodal_index", embedding_model, allow_dangerous_deserialization=True)
    return vector_store.as_retriever(search_kwargs={"k": 3})

def get_answer_from_llm(retriever, query_text=None, query_image=None):
    retrieval_input = query_text if query_text else Image.open(query_image)
    relevant_docs = retriever.invoke(retrieval_input)

    context_content = []
    st.write("---")
    st.markdown("#### 📚 검색된 관련 정보 (Context):")
    for doc in relevant_docs:
        if doc.metadata['type'] == 'text':
            context_content.append({"type": "text", "text": f"참고 텍스트: {doc.page_content}"})
            st.text(f"[텍스트] {doc.metadata['source']}")
        elif doc.metadata['type'] == 'image':
            image_uri = image_to_base64_uri(doc.metadata['path'])
            context_content.append({"type": "image_url", "image_url": {"url": image_uri}})
            st.image(doc.metadata['path'], caption=f"[이미지] {doc.metadata['source']}", width=200)

    prompt_text = "당신은 제미니 AI 연구소의 규정 안내 전문가입니다. 아래 '관련 정보'를 바탕으로 사용자의 '질문'에 대해 상세하고 친절하게 답변해주세요."
    
    user_original_query = [{"type": "text", "text": "--- 질문 ---"}]
    if query_text:
        user_original_query.append({"type": "text", "text": query_text})
    if query_image:
        image_uri = image_to_base64_uri(query_image)
        user_original_query.append({"type": "image_url", "image_url": {"url": image_uri}})

    final_prompt_content = [
        {"type": "text", "text": prompt_text},
        {"type": "text", "text": "--- 관련 정보 ---"},
        *context_content,
        *user_original_query
    ]
    
    llm = ChatVertexAI(model_name="gemini-1.0-pro-vision", location="asia-northeast3")
    message = HumanMessage(content=final_prompt_content)
    response = llm.invoke([message])
    return response.content

# --- Streamlit UI ---
st.set_page_config(page_title="멀티모달 RAG 챗봇", page_icon="🧠")
st.title("🧠 멀티모달 RAG 챗봇")
st.markdown("텍스트나 이미지를 사용하여 연구소 규정에 대해 질문해보세요.")

try:
    retriever = load_retriever()
    query_text = st.text_input("텍스트로 질문하기:")
    query_image = st.file_uploader("이미지로 질문하기:", type=["jpg", "jpeg", "png"])

    if st.button("답변 생성"):
        if not query_text and not query_image:
            st.warning("텍스트나 이미지를 입력해주세요.")
        else:
            with st.spinner("관련 정보를 찾고 Gemini가 답변을 생성하는 중입니다..."):
                if query_image:
                    st.image(query_image, caption="질문 이미지")
                answer = get_answer_from_llm(retriever, query_text, query_image)
                st.markdown("---")
                st.markdown("#### ✨ 최종 답변:")
                st.markdown(answer)
except FileNotFoundError:
    st.error("`faiss_multimodal_index` 인덱스 파일을 찾을 수 없습니다. 로컬에서 실행 중이라면, 먼저 `indexer.py`를 실행하여 인덱스를 생성해주세요. 클라우드 배포 시에는 인덱스 파일 업로드 방법을 고려해야 합니다.")
except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")