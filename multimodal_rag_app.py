# multimodal_rag_app.py (AttributeError 해결된 최종 버전)
import streamlit as st
import os
import base64
import mimetypes
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from PIL import Image

# --- 설정 ---
FAISS_INDEX_PATH = "faiss_multimodal_index"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "firestore-key.json"

# --- 헬퍼 함수: 이미지를 Base64로 변환 ---
def image_to_base64_uri(file_path_or_bytes):
    """이미지 파일 경로나 바이트 데이터를 Base64 데이터 URI로 변환합니다."""
    # 파일 경로인 경우 바이트로 읽기
    if isinstance(file_path_or_bytes, str):
        mime_type, _ = mimetypes.guess_type(file_path_or_bytes)
        if mime_type is None:
            mime_type = 'image/jpeg'
        with open(file_path_or_bytes, "rb") as image_file:
            bytes_data = image_file.read()
    # 바이트 데이터인 경우 그대로 사용 (UploadedFile 처리)
    else:
        # Streamlit UploadedFile은 .type 속성을 가짐
        mime_type = file_path_or_bytes.type
        bytes_data = file_path_or_bytes.getvalue()

    encoded_string = base64.b64encode(bytes_data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"

# --- 핵심 함수 ---
@st.cache_resource
def load_retriever():
    """로컬에 저장된 FAISS 인덱스를 로드하여 retriever를 생성합니다."""
    embedding_model = VertexAIEmbeddings(model_name="multimodalembedding@001")
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    return vector_store.as_retriever(search_kwargs={"k": 3})

def get_answer_from_llm(retriever, query_text=None, query_image=None):
    """텍스트 또는 이미지 쿼리를 사용하여 RAG를 수행하고 답변을 생성합니다."""
    # 1. 쿼리 임베딩을 위한 입력 구성
    retrieval_input = query_text if query_text else Image.open(query_image)

    # 2. RAG 수행: 벡터 DB에서 관련 문서 검색
    relevant_docs = retriever.invoke(retrieval_input)

    # 3. LLM에 전달할 컨텍스트 구성
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

    # 4. LLM 프롬프트 구성
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

    # 5. LLM 호출
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
    st.error(f"'{FAISS_INDEX_PATH}' 인덱스 파일을 찾을 수 없습니다. `indexer.py`를 먼저 실행하여 인덱스를 생성해주세요.")
except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")