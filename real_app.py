import streamlit as st
from PIL import Image
import glob
import os
import cv2

# selectbox 설정
sidebar_selectbox = st.sidebar.selectbox('Information', ['Introduction', 'Input Image', 'Output Image'])

# Home Page
if sidebar_selectbox == "Introduction":
    st.title(" Hello Virtual Fitting Room")
    
# input image 를 로컬저장소 dataset 폴더 안에 넣기
if sidebar_selectbox == "Input Image":
    st.subheader("Input Your Image")
    uploaded_clothes = st.file_uploader('이미지 파일을 올려주세요.', type=["png","jpg","jpeg"], key=1)
    uploaded_body = st.file_uploader('이미지 파일을 올려주세요.', type=["png","jpg","jpeg"], key=2)

    if uploaded_clothes is not None:
    
        img_1 = uploaded_clothes.name

        with open(os.path.join('C:/Users/shinj/Desktop/PF-AFN/PF-AFN_test/dataset/preprocessing/test_clothes', img_1), 'wb') as f:
            f.write(uploaded_clothes.read())

    if uploaded_body is not None:
    
        img_2 = uploaded_body.name

        with open(os.path.join('C:/Users/shinj/Desktop/PF-AFN/PF-AFN_test/dataset/preprocessing/test_img', img_2), 'wb') as f:
            f.write(uploaded_body.read())


# test.py 모델 
os.system("python test.py --name demo --resize_or_crop None --batchSize 1 --gpu_ids 0")

# model에서 나온 output image 출력
if sidebar_selectbox == "Output Image":
    st.subheader("Image")
    result_image = Image.open('C:/Users/shinj/Desktop/PF-AFN/PF-AFN_test/results/demo/PFAFN/*.jpg')
    st.image(result_image ,width=750)

