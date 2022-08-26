import streamlit as st
from PIL import Image
import glob
import os
import cv2
from rembg import remove

# selectbox 설정
sidebar_selectbox = st.sidebar.selectbox('Information', ['Introduction', 'Input Image', 'Output Image'])

# Home Page
if sidebar_selectbox == "Introduction":
    st.title(" Hello Virtual Fitting Room")
    
# input image 를 로컬저장소 dataset 폴더 안에 넣기
if sidebar_selectbox == "Input Image":
    
    # 파일 초기화
    print("file init")
    [os.remove(f) for f in glob.glob('dataset/bf_test_img/*')]
    [os.remove(f) for f in glob.glob('dataset/test_img/*')]
    [os.remove(f) for f in glob.glob('dataset/test_edge/*')]
    [os.remove(f) for f in glob.glob('dataset/test_clothes/*')]
    [os.remove(f) for f in glob.glob('results/demo/PFAFN/*')]

    st.subheader("Input Your Image")
    uploaded_clothes = st.file_uploader('Input Clothes', type=["png","jpg","jpeg"], key=1)
    uploaded_body = st.file_uploader('Input Body Image', type=["png","jpg","jpeg"], key=2)

    if uploaded_clothes is not None:
    
        img_1 = uploaded_clothes.name

        with open(os.path.join('C:/Users/shinj/Desktop/PF-AFN/PF-AFN_test/dataset/test_clothes', img_1), 'wb') as f:
            f.write(uploaded_clothes.read())

    if uploaded_body is not None:
    
        img_2 = uploaded_body.name

        with open(os.path.join('C:/Users/shinj/Desktop/PF-AFN/PF-AFN_test/dataset/bf_test_img', img_2), 'wb') as f:
            f.write(uploaded_body.read())

# 로컬폴더에 받은 사진 파일을 불러들임
print('preprocessing_start')
bf_human_pathes = glob.glob('dataset/bf_test_img/*')
human_pathes = glob.glob('dataset/test_img/*')

cloth_pathes = glob.glob('dataset/test_clothes/*')
# demo.txt 초기화
with open("./demo.txt", "w") as f:
    f.write("")
    f.close()

# input 된 전신 사진 1차 전처리 (remove background)
for human_path in bf_human_pathes:
    # dataset/bf_test_img/ => 20
    human_name = human_path[20:]
    
    img_human = cv2.imread(human_path)
    img_human = cv2.resize(img_human,(192,256))
    cv2.imwrite(human_path, img_human)
    
    with open(human_path , 'rb') as i:
        with open('dataset/test_img/'+human_name, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)

# # 2차 전처리(resizing cloth &  generating binary image)
for human_path in human_pathes:
    human_name = human_path[17:]
    for cloth_path in cloth_pathes:
        cloth_name = cloth_path[21:]
        img_cloth = cv2.imread(cloth_path) 
        # resize
        img_cloth = cv2.resize(img_cloth, (192,256))
        
        img_cloth_g = cv2.cvtColor(img_cloth,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img_cloth_g,(5,5),0)
        ret , cloth_binary = cv2.threshold(blur,230,255,cv2.THRESH_BINARY_INV)
        cv2.imwrite(cloth_path,img_cloth)
        cv2.imwrite('dataset/test_edge/'+cloth_name,cloth_binary)
#         with open("./demo.txt", "r") as f:
#             lines = f.readlines()
        with open("./demo.txt", "a") as f:
            f.write(human_name+" "+cloth_name+"\n")
            f.close()
print('preprocessing _end')
# test.py 모델 
# model에서 나온 output image 출력  
if sidebar_selectbox == "Output Image":
    print('model_start')  
    os.system("python test.py --name demo --resize_or_crop None --batchSize 1 --gpu_ids 0")
    print('model_end')  
    result_path = glob.glob('results/demo/PFAFN/*.jpg')[-1]
    result_name=result_path[19:]

    st.subheader("Image")
    result_image = Image.open(os.path.join('C:/Users/shinj/Desktop/PF-AFN/PF-AFN_test/results/demo/PFAFN/', result_name))
    st.image(result_image ,width=750)

