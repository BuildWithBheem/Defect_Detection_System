import tensorflow as tf #type: ignore
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array #type: ignore
from tf_keras_vis.gradcam import Gradcam #type: ignore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear #type:ignore
from tf_keras_vis.utils.scores import BinaryScore  #type:ignore
print(tf.__version__)
cnn = tf.keras.models.load_model(r"C:\Users\KIIT\OneDrive\Desktop\ML-DATASETS\Defect_detection\Defect_detection_system.keras")

image_jpg = load_img(r'C:\Users\KIIT\OneDrive\Desktop\ML-DATASETS\Defect_detection\cast_def_0_138.jpeg',target_size = (128,128))

image_jpg = img_to_array(image_jpg)
image_jpg = image_jpg/255                   # Normalisation
image_jpg = np.expand_dims(image_jpg,axis =0)
inputs = tf.keras.Input(shape=(128, 128, 3))
x = inputs
conv_output_tensor = None
for layer in cnn.layers:
    x = layer(x)
    if layer.name == "conv2d_2":
        conv_output_tensor = x

grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_output_tensor, x])
image_tensor = tf.convert_to_tensor(
    image_jpg,
    dtype=tf.float32
)

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(
        image_tensor,
        training=False
    )

    tape.watch(conv_outputs) 
    pred_index = tf.argmax(predictions[0])

    class_channel = predictions[:, pred_index]

    grads = tape.gradient(
    class_channel,
    conv_outputs
)
    print(type(grads))

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0, 1, 2)
)

print(pooled_grads.shape)

heatmap = tf.reduce_sum(
    conv_outputs * pooled_grads,
    axis=-1
)

print(heatmap.shape)

heatmap = tf.maximum(heatmap, 0)

heatmap /= tf.reduce_max(heatmap)

heatmap = heatmap.numpy()

import cv2

heatmap = tf.squeeze(heatmap)

print(heatmap.shape)

heatmap = tf.image.resize(
    heatmap[..., tf.newaxis],
    (128, 128)
)

heatmap = tf.squeeze(heatmap)

heatmap = heatmap.numpy()

print(heatmap.shape)

heatmap = cv2.resize(
    heatmap,
    (128, 128)
)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_jpg[0])
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image_jpg[0])
plt.imshow(heatmap, cmap="jet", alpha=0.45)
plt.title("Grad-CAM")
plt.axis("off")

plt.show()