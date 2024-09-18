import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#
# 사진한장 뽑아보기
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()


# 신경망 범위를 0 ~ 1 로 소숫점화 하여 나타내기 위하여 255로 나누어줌
# ex) 0.125031, 0.855673
train_images = train_images / 255.0
test_images = test_images / 255.0
#


# plt.figure(figsize=(10,10))
# for i in range(25):
#    plt.subplot(5,5,i+1)                                    # 여러개의 그래프를 하나의 그림에 표기
#    plt.xticks([])                                          # x축 눈금
#    plt.yticks([])                                          # y축 눈금
#    plt.grid(False)                                         # 격자 비활성화
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])                # x축 레이블 설정
# plt.show()



# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),          # tf.keras.layers.Flatten이 28*28픽셀의 2차원배열을 784픽셀의 1차원배열화  
#     tf.keras.layers.Dense(128, activation='relu'),          
#     tf.keras.layers.Dense(10)
# ])

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=10)


# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
# print('\nTest accuracy:', test_acc)

img = test_images[1]
print(img.shape)


img = (np.expand_dims(img,0))
print(img.shape)