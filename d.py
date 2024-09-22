import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Sequential 모델 생성
model = models.Sequential()

# 사전 학습된 VGG16 모델을 Sequential 모델의 첫 번째 레이어로 추가
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# VGG16의 기존 가중치를 고정하여 학습되지 않도록 설정
for layer in base_model.layers:
    layer.trainable = False

# Sequential 모델에 base_model 추가
model.add(base_model)

# Flatten 레이어와 Dense 레이어 추가
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))  # 이진 분류

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 데이터 전처리 (ImageDataGenerator를 이용한 데이터 증강)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 학습 데이터의 20%를 검증 데이터로 사용
)

# 학습 데이터 로더 (subset='training' 옵션 사용)
train_generator = train_datagen.flow_from_directory(
    './image_data/',  # 데이터 경로
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'  # 학습용 데이터 (80%)
)

# 검증 데이터 로더 (subset='validation' 옵션 사용)
validation_generator = train_datagen.flow_from_directory(
    './image_data/',  # 동일한 경로에서 20%를 검증 데이터로 사용
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # 검증용 데이터 (20%)
)

# 모델 학습
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# 학습 결과 평가
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
