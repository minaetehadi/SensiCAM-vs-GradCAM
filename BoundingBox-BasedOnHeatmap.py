import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import requests

# Load a pre-trained model
model = models.resnet101(weights='DEFAULT')
model.eval()

# Load ImageNet class labels
def load_imagenet_labels(url='https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'):
    response = requests.get(url)
    return response.json()

imagenet_labels = load_imagenet_labels()

# Load and preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_path = "/content/Lion.jpg"
image = Image.open(image_path)
image_tensor = transform(image).unsqueeze(0)

# Function to compute Grad-CAM
def generate_grad_cam(feature_map, grad_weights):
    size_upsample = (224, 224)
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)

    for i, w in enumerate(grad_weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)  # Apply ReLU to the CAM
    cam = cv2.resize(cam, size_upsample)
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam

def grad_cam(model, image, target_class):
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fw = model.layer4[-1].register_forward_hook(forward_hook)
    handle_bw = model.layer4[-1].register_backward_hook(backward_hook)

    output = model(image)
    model.zero_grad()

    one_hot_output = torch.zeros((1, output.size()[-1]), dtype=torch.float32)
    one_hot_output[0][target_class] = 1
    output.backward(gradient=one_hot_output)

    handle_fw.remove()
    handle_bw.remove()

    grads_val = gradients[0].cpu().data.numpy()[0]
    target = features[0].cpu().data.numpy()[0]

    grad_weights = np.mean(grads_val, axis=(1, 2))

    cam = generate_grad_cam(target, grad_weights)

    return cam, grads_val

# Function to calculate bounding box based on 50% max value of heatmap
def get_bounding_box(heatmap, threshold=0.5):
    """Returns the bounding box for regions in the heatmap that are above the given threshold."""
    max_val = np.max(heatmap)
    threshold_value = max_val * threshold
    binary_map = heatmap > threshold_value

    # Find the bounding box coordinates
    y_indices, x_indices = np.where(binary_map)
    if y_indices.size == 0 or x_indices.size == 0:
        return None  # No region found

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    return x_min, y_min, x_max, y_max

def mixcam_attack_untargeted(model, image, epsilon, num_iterations, decay_factor, fusion_ratio):
    alpha = epsilon / num_iterations
    perturbed_image = image.clone()
    momentum = torch.zeros_like(image)
    gradients_list = []

    for i in range(num_iterations):
        perturbed_image.requires_grad_()

        output = model(perturbed_image)
        pred_class = output.argmax().item()

        cam, gradient = grad_cam(model, perturbed_image, pred_class)
        gradients_list.append(gradient)

        # Mask values lower than 50% of the max value in the heatmap
        max_val = np.max(cam)
        mask = cam >= (0.5 * max_val)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(image.device)

        masked_image = image * mask
        mixed_image = fusion_ratio * image + (1 - fusion_ratio) * masked_image
        mixed_image = Variable(mixed_image, requires_grad=True)

        output = model(mixed_image)
        loss = F.cross_entropy(output, torch.LongTensor([pred_class]).to(image.device))
        model.zero_grad()
        loss.backward()

        gradient = mixed_image.grad.data
        momentum = decay_factor * momentum + gradient / torch.norm(gradient, p=1)
        perturbed_image = perturbed_image + alpha * torch.sign(momentum)

        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image = perturbed_image.detach()

    # Calculate the bounding box based on 50% of the max value of the heatmap
    bounding_box = get_bounding_box(cam)

    return perturbed_image, pred_class, cam, mask, masked_image, mixed_image, gradients_list, bounding_box

# Parameters
epsilon = 0.05
num_iterations = 5
decay_factor = 1.0
fusion_ratio = 0.6

# Generate adversarial example
perturbed_image, pred_class, cam, mask, masked_image, mixed_image, gradients_list, bounding_box = mixcam_attack_untargeted(
    model, image_tensor, epsilon, num_iterations, decay_factor, fusion_ratio
)

# Convert tensors to numpy arrays for visualization
def tensor_to_np(tensor):
    return tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)

original_image_np = tensor_to_np(image_tensor)
perturbed_image_np = tensor_to_np(perturbed_image)
masked_image_np = tensor_to_np(masked_image)
mixed_image_np = tensor_to_np(mixed_image)

# Convert momentum tensor to numpy array for visualization
def compute_alpha_sign_momentum(alpha, momentum_tensor):
    return alpha, momentum_tensor, alpha * torch.sign(momentum_tensor).squeeze().detach().cpu().numpy().transpose(1, 2, 0)

momentum_tensor = torch.zeros_like(image_tensor)
alpha, momentum_tensor, momentum_image = compute_alpha_sign_momentum(epsilon / num_iterations, momentum_tensor)

# Save the original image for overlay
def save_image_from_np(np_image, filename):
    """ Save numpy image array to file """
    cv2.imwrite(filename, np.uint8(np_image * 255))

# Save the original image to overlay heatmap
save_image_from_np(original_image_np, 'original_image.jpg')

# Overlay heatmaps
def overlay_heatmap(img_path, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
    img = cv2.imread(img_path)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay

# Overlay Grad-CAM on the perturbed image
overlay_img_with_heatmap = overlay_heatmap('original_image.jpg', cam, alpha=0.6)

# Draw bounding box on the overlayed image
cropped_image = None
if bounding_box is not None:
    x_min, y_min, x_max, y_max = bounding_box
    cv2.rectangle(overlay_img_with_heatmap, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

    # Extract and display the cropped region from the original image
    cropped_image = original_image_np[y_min:y_max, x_min:x_max]

# Convert BGR image to RGB for display
overlay_img_with_heatmap_rgb = cv2.cvtColor(overlay_img_with_heatmap, cv2.COLOR_BGR2RGB)

# Show the images and matrices
fig, ax = plt.subplots(2, 3, figsize=(25, 15))

# Original Image
ax[0, 0].imshow(original_image_np)
ax[0, 0].set_title("Original Image")
ax[0, 0].axis('off')

# Grad-CAM Heatmap
ax[0, 1].imshow(cam, cmap='jet')
ax[0, 1].set_title("Grad-CAM Heatmap")
ax[0, 1].axis('off')

# Overlay Grad-CAM on perturbed image
ax[1, 0].imshow(overlay_img_with_heatmap_rgb)
ax[1, 0].set_title("Overlay Heatmap with Bounding Box")
ax[1, 0].axis('off')

# Masked Image
ax[1, 1].imshow(masked_image_np)
ax[1, 1].set_title("Masked Image")
ax[1, 1].axis('off')

# Cropped Region (if bounding box is found)
if cropped_image is not None:
    ax[0, 2].imshow(cropped_image)
    ax[0, 2].set_title("Cropped Region (Original Image)")
    ax[0, 2].axis('off')

plt.show()
