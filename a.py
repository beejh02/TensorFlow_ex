import tensorflow as tf

# 이미지 크기 및 배치 크기 설정
img_height = 180
img_width = 180
batch_size = 32

# 학습용 데이터셋 로드
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "C:/Users/suk01/Desktop/Programming/TensorFlow_ex/image_data/",  # 이미지 폴더 경로
  validation_split=0.2,  # 20%는 검증용 데이터로 사용
  subset="training",  # 학습용 데이터로 설정
  seed=123,  # 랜덤 시드 고정
  image_size=(img_height, img_width),  # 이미지 크기 조정
  batch_size=batch_size  # 배치 크기 설정
)

# 검증용 데이터셋 로드
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "C:/Users/suk01/Desktop/Programming/TensorFlow_ex/image_data/",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)  # 예시 출력: ['cats', 'dogs']


######################
######################


from tensorflow.keras import layers

# 데이터 증강 레이어
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
])


AUTOTUNE = tf.data.AUTOTUNE

# 학습 데이터셋 캐시 및 프리페치
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# 검증 데이터셋 캐시 및 프리페치
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



model = tf.keras.Sequential([
  data_augmentation,  # 데이터 증강 레이어 추가
  layers.Rescaling(1./255),  # 이미지 값을 0~1 사이로 정규화
  layers.Conv2D(32, 3, activation='relu'),  # 첫 번째 Convolution 층
  layers.MaxPooling2D(),  # 풀링 층
  layers.Conv2D(64, 3, activation='relu'),  # 두 번째 Convolution 층
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, activation='relu'),  # 세 번째 Convolution 층
  layers.MaxPooling2D(),
  layers.Flatten(),  # 2D 데이터를 1D로 변환
  layers.Dense(128, activation='relu'),  # Fully Connected Layer
  layers.Dense(len(class_names), activation='softmax')  # 클래스 수에 맞춰 출력층 설정
])

model.summary()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


loss, accuracy = model.evaluate(val_ds)
print(f"Validation accuracy: {accuracy}")


import numpy as np
from tensorflow.keras.preprocessing import image

# 새로운 이미지 로드 및 전처리
img = image.load_img('C:/Users/suk01/Desktop/Programming/TensorFlow_ex/검증이미지/찌그러진캔.jpg', target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # 배치 차원 추가

# 예측 수행
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
print(f"Predicted class: {class_names[predicted_class]}")
