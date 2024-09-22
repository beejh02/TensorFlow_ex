import os

train_dir = './image_data/'
validation_dir = './Classification_Image/'

# 학습 데이터 경로에 있는 파일 목록 출력
print(os.listdir(train_dir))

# 검증 데이터 경로에 있는 파일 목록 출력
print(os.listdir(validation_dir))
