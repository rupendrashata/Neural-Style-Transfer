import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL

from tensorflow.keras import Model


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
print(tf.test.gpu_device_name())
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# vgg = tf.keras.applications.VGG19(include_top=True, weights="imagenet")
#
# for layers in vgg.layers:
#     print(f"{layers.name} --> {layers.output_shape}")


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    gram = tf.expand_dims(result, axis=0)
    input_shape = tf.shape(input_tensor)
    i_j = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return gram/i_j


def load_vgg():
    vgg = tf.keras.applications.VGG19(include_top=True, weights="imagenet")
    vgg.trainable = False
    content_layers = ['block4_conv2']
    style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
    content_output = vgg.get_layer(content_layers[0]).output
    style_outputs = [vgg.get_layer(style_layer).output for style_layer in style_layers]

    gram_style = [gram_matrix(style_output) for style_output in style_outputs]

    model = Model([vgg.input], [content_output, gram_style])
    return model


content_image = cv2.resize(cv2.imread('content_image.png'), (224, 224))
content_image = tf.image.convert_image_dtype(content_image, tf.float32)
style_image = cv2.resize(cv2.imread('style_image.png'), (224, 224))
style_image = tf.image.convert_image_dtype(style_image, tf.float32)
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(np.array(content_image), cv2.COLOR_BGR2RGB))
# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(np.array(style_image), cv2.COLOR_BGR2RGB))
# plt.show()

opt = tf.optimizers.Adam(learning_rate=0.01, beta_1=0.99, epsilon=1e-1)


def loss_object(style_outputs, content_outputs, style_target, content_target):
    style_weight = 1e-2
    content_weight = 1e-1
    content_loss = tf.reduce_mean((content_outputs - content_target)**2)
    style_loss = tf.add_n([tf.reduce_mean((output_ - target_)**2) for output_, target_ in zip(style_outputs, style_target)])
    total_loss = content_weight*content_loss + style_weight*style_loss
    return total_loss


vgg_model = load_vgg()
content_target = vgg_model(np.array([content_image*255]))[0]
style_target = vgg_model(np.array([style_image*255]))[1]


def train_step(image, epoch):
  with tf.GradientTape() as tape:
    output = vgg_model(image*255)
    loss = loss_object(output[1], output[0], style_target, content_target)
  gradient = tape.gradient(loss, image)
  opt.apply_gradients([(gradient, image)])
  image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))
  tf.print(f"Loss = {loss}")


EPOCHS = 20
image = tf.image.convert_image_dtype(content_image, tf.float32)
image = tf.Variable([image])
for i in range(EPOCHS):
    train_step(image, i)

tensor = image*255
tensor = np.array(tensor, dtype=np.uint8)
if np.ndim(tensor)>3:
  assert tensor.shape[0] == 1
  tensor = tensor[0]
tensor =  PIL.Image.fromarray(tensor)
plt.imshow(cv2.cvtColor(np.array(tensor), cv2.COLOR_BGR2RGB))
plt.show()










