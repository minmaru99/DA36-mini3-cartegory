import streamlit as st
import streamlit as st
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import base64
import time

if 'page' not in st.session_state:
    st.session_state['page'] = 1


# 페이지 변경 함수
def go_to_page(page_num):
    st.session_state['page'] = page_num


if st.session_state['page'] == 1:

    main_bg_color = "#ECF8E0"  # 메인 페이지 배경색
    # CSS 스타일을 적용하여 배경 색 변경
    st.markdown(f"""
        <style>
        /* 메인 페이지 배경 색 설정 */
        .stApp {{
            background-color: {main_bg_color};
        }}
        /* 제목 스타일 */
        .big-title {{
            font-size: 4em;  /* 글자 크기 조정 */
            text-align: center;
            margin-bottom: 0.5em;
            color: #333;  /* 제목 색상 */
        }}
        .big-subtitle {{
            font-size: 1em;  /* 부제목 크기 조정 */
            text-align: center;
            margin-bottom: 1.5em;
            color: #666;  /* 부제목 색상 */
        }}
        /* 이미지 크기 조정 */
        .custom-image {{
            display: block;
            margin: 0 auto;  /* 이미지 중앙 정렬 */
            width: 80%;  /* 이미지 너비 조정 (원하는 크기로 설정) */
            max-width: 600px;  /* 최대 너비 제한 */
            height: auto;  /* 비율 유지 */
        }}
        </style>
        """, unsafe_allow_html=True)

    # 이미지 출력 (크기 조정 및 중앙 정렬)
    st.image('data/park.jpg')
    # 큰 제목 한 글자씩 출력 (누적)
    title = "Where to Park?"
    subtitle = "CARTEGORY BY 하와수"

    title_placeholder = st.empty()
    current_title = ""
    for char in title:
        current_title += char
        title_placeholder.markdown(f"<div class='big-title'>{current_title}</div>", unsafe_allow_html=True)
        time.sleep(0.2)

    st.markdown(f"<div class='big-subtitle'>{subtitle}</div>", unsafe_allow_html=True)

    st.markdown('-' * 10)
    col1, col2, col3 = st.columns([2, 2, 2])
    with col3:
        # 버튼 출력
        if st.button("LeT's gO 🏃🏻‍♂️🏃🏻‍♂️"):
            # 로딩 진행 표시
            progress_bar = st.progress(0)  # 진행 상태를 나타낼 막대 생성
            for percent_complete in range(100):
                time.sleep(0.01)  # 진행 상태를 갱신하기 위해 잠시 대기
                progress_bar.progress(percent_complete + 1)  # 진행 상태 갱신
            # 페이지 전환
            st.session_state['page'] = 2
            st.rerun()  # 페이지 리프레시


elif st.session_state['page'] == 2:
    st.markdown(
        """
        <style>
        .stApp {
            max-width: 140%;  /* 페이지의 가로 너비를 100%로 설정 */
            padding-top: 50px;  /* 상단 여백 */
        }
        </style>
        """, unsafe_allow_html=True
    )
    image_path = "data/parkinglot.png"  # 배경 이미지 경로
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
            <style>
            .stApp {{
                background: url(data:image/png;base64,{base64_image});
                background-size: cover;
            }}
            </style>
            """,
        unsafe_allow_html=True
    )

    st.markdown('<br>' * 7, unsafe_allow_html=True)
    st.title('Please register your vehicle 👀')
    st.write('-' * 10)
    st.subheader('Upload here⬇️')


    # 모델 로드
    @st.cache_resource  # 캐싱을 통해 모델 로드 속도 향상
    def load_trained_model(model_path):
        return load_model(model_path)


    # model_path = "mb_model.h5"
    model_path = "Xception_model.h5"
    model = load_trained_model(model_path)

    # 차량 모델 리스트
    label_classes = ['Carens', 'Kona', 'Mohave', 'Niro', 'Palisade', 'Santafe',
                     'Seltos', 'Sorento', 'Soul', 'Sportage', 'Tucson', 'Veracruz']

    # 현대차, 기아차 모델 리스트
    electric_vehicles = ['Kona', 'Niro', 'Model', 'Santefe', 'Palisade', 'Soul']
    hyundai_models = ['Palisade', 'Tucson', 'Santafe', 'Veracruz', 'Kona', 'Niro']
    kia_models = ['Carens', 'Mohave', 'Seltos', 'Sorento', 'Soul', 'Sportage']

    # 이미지 업로드
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
    # col1, col2, col3 = st.columns([1, 2, 1])
    # with col2:
    #     with st.container()
    with st.container():
        col1, col2 = st.columns([1, 1])  # 두 개의 열로 나누기
        with col1:
            if uploaded_file is not None:
                # 업로드된 이미지 표시
                st.image(uploaded_file, caption="", use_column_width=True)

        with col2:
            if uploaded_file is None:
                st.markdown('')
            if uploaded_file is not None:
                # 이미지 전처리
                IMAGE_SIZE = 299  # 모델 입력 크기
                image = Image.open(uploaded_file).convert("RGB")  # 이미지를 RGB로 변환
                image = image.resize((IMAGE_SIZE, IMAGE_SIZE))  # 크기 조정
                image_array = np.array(image)
                image_array = preprocess_input(image_array)  # 모델 입력 형식에 맞게 전처리
                batch_image = np.expand_dims(image_array, axis=0)  # 배치 차원 추가

                # 모델 예측
                pred_proba = model.predict(batch_image)
                pred_index = np.argmax(pred_proba)
                pred_label = label_classes[pred_index]
                confidence = pred_proba[0][pred_index]

                st.session_state.pred_label = pred_label
                st.session_state.confidence = confidence

                with st.expander("Image's info"):
                    st.write(f"**파일명:** {uploaded_file.name}")
                    st.write(f"**원본 크기:** {image.size}")
                    st.write(f"**데이터 타입:** {type(image)}")
                    # st.write(f"**형태:** {image.shape}")

                if confidence > 0.5:
                    if pred_label in hyundai_models:
                        st.markdown(
                            f"""
                                <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; font-size: 20px; color: #155724; margin-bottom: 10px;">
                                    🏭... Manufacturer is <strong>Hyundai</strong>
                                </div>
                                <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; font-size: 20px; color: #155724; margin-bottom: 10px;">
                                    🚗... Your car is <strong>{pred_label}</strong>
                                </div>
                                <div style="background-color: #CEECF5; padding: 10px; border-radius: 5px; font-size: 20px; color: #08298A;">
                                    🤖... Accuracy is <strong>{confidence * 100:.2f}%</strong>
                                </div>
                                """, unsafe_allow_html=True
                        )
                    elif pred_label in kia_models:
                        st.markdown(
                            f"""
                                <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; font-size: 20px; color: #155724; margin-bottom: 10px;">
                                    🏭... Manufacturer is <strong>Kia</strong>
                                </div>
                                <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; font-size: 20px; color: #155724; margin-bottom: 10px;">
                                    🚗... Your car is <strong>{pred_label}</strong>
                                </div>
                                <div style="background-color: #CEECF5; padding: 10px; border-radius: 5px; font-size: 20px; color: #08298A;">
                                    🤖... Accuracy is <strong>{confidence * 100:.2f}%</strong>
                                </div>
                                """, unsafe_allow_html=True
                        )

                    SAVE_DIR = "./uploaded_images"
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    save_path = os.path.join(SAVE_DIR, uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                else:
                    st.markdown('<br>' * 5, unsafe_allow_html=True)
                    st.markdown(
                        f"""
                            <div style="background-color: #F6CECE; padding: 8px; border-radius: 5px; font-size: 20px; color: #DF0101; margin-bottom: 8px; width: 300px;">
                                Sorry, I couldn't recognize this..
                            </div>
                            """, unsafe_allow_html=True
                    )

    if 'pred_label' in st.session_state and 'confidence' in st.session_state:
        pred_label = st.session_state.pred_label
        confidence = st.session_state.confidence

        if uploaded_file is None:
            st.markdown('')
        else:
            if confidence > 0.5:
                if pred_label in electric_vehicles:
                    # 전기차 안내 문구
                    st.markdown(
                        f"<h3 style='font-size: 25px; color: black;'>잠깐⚠️ {pred_label}는 전기차입니다. 지상 주차장을 이용해주세요.</h3>",
                        unsafe_allow_html=True)
                else:
                    # 지하 주차장 안내 문구
                    st.markdown(
                        f"<h3 style='font-size: 25px; color: black;'>야호! {pred_label} 은 지하 주차장에 진입할 수 있습니다. 🥳</h3>",
                        unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='font-size: 25px; color: black;'>엥? 차 사진을 등록해주세요;</h3>",
                            unsafe_allow_html=True)

    st.write('-' * 10)
    # st.image('data/info.jpg')
    col1, col2, col3 = st.columns([2, 4, 2])  # 좌측, 중앙, 우측 열로 나누기
    with col1:
        if st.button("⬅️Back"):
            go_to_page(1)

    with col3:
        if st.button("More➡️"):
            go_to_page(3)


elif st.session_state['page'] == 3:
    main_bg_color = "#ECF8E0"  # 메인 페이지 배경색
    st.markdown(f"""
        <style>
        /* 메인 페이지 배경 색 설정 */
        .stApp {{
            background-color: {main_bg_color};
        }}
        </style>
        """, unsafe_allow_html=True)

    link = "https://www.chargekorea.com/charge/index.php?"  # 이동할 링크

    main_bg_color = "#ECF8E0"  # 메인 페이지 배경색
    st.markdown(f"""
            <style>
            /* 메인 페이지 배경 색 설정 */
            .stApp {{
                background-color: {main_bg_color};
            }}
            </style>
            """, unsafe_allow_html=True)

    st.title("EV charging stations near you!")
    st.markdown('<br>', unsafe_allow_html=True)
    image_url = "https://www.chargekorea.com/charge/index.php?"  # 원하는 링크
    image_path = "data/evicon.png"  # 이미지 경로

    # HTML 코드로 이미지를 링크로 감싸기
    st.markdown(
        f"""
        <a href="{image_url}" target="_blank">
            <img src="data:image/png;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}" alt="Click here" width="500"/>
        </a>
        <p style="text-align:left; font-size:17px; color:#888;">(Click the image to go to the website.)</p>
        """, unsafe_allow_html=True
    )

    # st.title("EV charging stations near you!")
    # st.write('Click the image to go to the website.')
    st.write('-' * 20)

    col1, col2, col3 = st.columns([2, 4, 2])  # 좌측, 중앙, 우측 열로 나누기

    with col1:
        if st.button("🏠 Home"):
            go_to_page(1)

