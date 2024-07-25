#Finding Bounding Box From Heatmap

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import tensorflow as tf

# Load and preprocess the image
def load_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Display image
def display_image(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

img_path = '/content/golden.jpeg'  # Update with the path to your image
img_array = load_image(img_path)

# Load VGG16 model
model = VGG16(weights='imagenet')

# Set the class index to the predicted class
predictions = model.predict(img_array)
class_idx = np.argmax(predictions[0])
class_label = decode_predictions(predictions)[0][0][1]
print(f'Using class index: {class_idx}')
print(f'Class label: {class_label}')

# Define the model for Sensi-CAM
last_conv_layer = model.get_layer('block5_conv3')
heatmap_model = Model([model.inputs], [last_conv_layer.output, model.output])

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

# Overlay heatmap on image
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

overlay_heatmap(img_path, sensi_heatmap, f'Sensi-CAM: {class_label} (class {class_idx})')

# Find bounding box from heatmap
def find_bounding_box(heatmap, threshold=0.2):
    heatmap = cv2.resize(heatmap, (224, 224))
    binary_map = np.uint8(heatmap > threshold * heatmap.max())
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    x, y, w, h = cv2.boundingRect(contours[0])
    return x, y, w, h

x, y, w, h = find_bounding_box(sensi_heatmap)
print(f'Bounding box coordinates: x={x}, y={y}, w={w}, h={h}')

# Crop the most important part of the image
def crop_important_part(img_path, bbox):
    img = cv2.imread(img_path)
    x, y, w, h = bbox
    cropped_img = img[y:y+h, x:x+w]
    return cropped_img

cropped_img = crop_important_part(img_path, (x, y, w, h))
display_image(cropped_img)

# Save the cropped image
cropped_img_path = '/path/to/save/cropped_image.jpg'  # Update with your desired save path
cv2.imwrite(cropped_img_path, cropped_img)
print(f'Cropped image saved to {cropped_img_path}')
