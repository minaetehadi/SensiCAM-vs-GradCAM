# Comparing Grad-CAM vs SensiCAM 

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D

# Load and preprocess the image
def load_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Display image
def display_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

img_path = '/path/image.jpg'
img_array = load_image(img_path)
display_image(img_path)

# Load VGG16 model
model = VGG16(weights='imagenet')

# Set the class index to 162 and get the class name
class_idx = 162
class_label = decode_predictions(np.eye(1, 1000, class_idx))[0][0][1]
print(f'Using class index: {class_idx}')
print(f'Class label: {class_label}')

# Define the model for Grad-CAM
last_conv_layer = model.get_layer('block5_conv3')
heatmap_model = Model([model.inputs], [last_conv_layer.output, model.output])

with tf.GradientTape() as tape:
    conv_output, predictions = heatmap_model(img_array)
    output = predictions[:, class_idx]

grads = tape.gradient(output, conv_output)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_output = conv_output[0].numpy()
pooled_grads = pooled_grads.numpy()

for i in range(pooled_grads.shape[-1]):
    conv_output[:, :, i] *= pooled_grads[i]

heatmap = np.mean(conv_output, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# Function to compute Sensi-CAM heatmap
def sensi_cam_heatmap(model, img_array, class_idx):
    with tf.GradientTape() as tape:
        conv_output, predictions = heatmap_model(img_array)
        output = predictions[:, class_idx]
    grads = tape.gradient(output, conv_output)
    grads = grads[0].numpy()
    conv_output = conv_output[0].numpy()
    sensitivity_map = np.sum((grads * conv_output), axis=-1)
    sensitivity_map = np.maximum(sensitivity_map, 0)
    sensitivity_map /= np.max(sensitivity_map)
    return sensitivity_map

sensi_heatmap = sensi_cam_heatmap(model, img_array, class_idx)

# Function to overlay heatmap on image
def overlay_heatmap(img_path, heatmap, title, alpha=0.4, colormap=cv2.COLORMAP_JET):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to display 3D histogram of heatmap
def plot_3d_histogram(heatmap, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(0, heatmap.shape[1] - 1, heatmap.shape[1])
    y = np.linspace(0, heatmap.shape[0] - 1, heatmap.shape[0])
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, heatmap, cmap='viridis')
    plt.title(title)
    plt.show()

# Display heatmaps without overlay
print("Grad-CAM Heatmap")
plt.imshow(heatmap, cmap='viridis')
plt.title(f'Grad-CAM: {class_label} (class {class_idx})')
plt.axis('off')
plt.show()

print("Sensi-CAM Heatmap")
plt.imshow(sensi_heatmap, cmap='viridis')
plt.title(f'Sensi-CAM: {class_label} (class {class_idx})')
plt.axis('off')
plt.show()

# Display heatmaps with overlay
overlay_heatmap(img_path, heatmap, f'Grad-CAM: {class_label} (class {class_idx})')
overlay_heatmap(img_path, sensi_heatmap, f'Sensi-CAM: {class_label} (class {class_idx})')

# Display 3D histograms of heatmaps
plot_3d_histogram(heatmap, f'Grad-CAM 3D Histogram: {class_label} (class {class_idx})')
plot_3d_histogram(sensi_heatmap, f'Sensi-CAM 3D Histogram: {class_label} (class {class_idx})')
