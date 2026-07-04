import tensorflow as tf #type: ignore
import numpy as np
import matplotlib.pyplot as plt

def generate_gradcam(cnn,image_jpg):
    inputs = tf.keras.Input(shape=(128,128,3))
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

    heatmap = cv2.resize(
    heatmap,
    (128, 128)
    )

    # Convert heatmap to uint8
    heatmap = np.uint8(255 * heatmap)

    # Apply JET color map
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Original image
    og_img = (image_jpg[0] * 255).astype(np.uint8)

    # OpenCV expects RGB
    original = cv2.cvtColor(
        og_img,
        cv2.COLOR_RGB2BGR
    )

    # Overlaying ( Overlap defect areas on original image)
    overlay = cv2.addWeighted(
        og_img,
        0.6,
        heatmap,
        0.4,
        0
    )

    success, buffer = cv2.imencode(".png", overlay)

    if not success:
        raise ValueError("Failed to encode heatmap.")

    return buffer.tobytes()
