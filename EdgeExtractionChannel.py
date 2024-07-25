import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(image_path):
    # Read the image from the file path
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect.")
    return image

def sobel_edge_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel filter to find edges
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude of the gradient
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # Convert to uint8 (0-255 range)
    sobel_mag = np.uint8(255 * sobel_mag / np.max(sobel_mag))

    return sobel_mag

def canny_edge_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detector
    edges = cv2.Canny(gray, 100, 200)

    return edges

def combine_edges(sobel_edges, canny_edges):
    # Combine Sobel and Canny edges
    combined_edges = cv2.addWeighted(sobel_edges, 0.5, canny_edges, 0.5, 0)

    # Convert single channel to 3 channels
    combined_edges = cv2.cvtColor(combined_edges, cv2.COLOR_GRAY2BGR)

    return combined_edges

def superimpose_images(original, edges):
    # Superimpose the original image with the edge-detected image
    superimposed = cv2.addWeighted(original, 0.5, edges, 0.5, 0)

    return superimposed

def display_images(original, sobel_edges, canny_edges, combined_edges, superimposed):
    # Display the images using matplotlib
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.title("Sobel Edge Detection")
    plt.imshow(sobel_edges, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.title("Canny Edge Detection")
    plt.imshow(canny_edges, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.title("Combined Edges")
    plt.imshow(cv2.cvtColor(combined_edges, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.title("Superimposed Image")
    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Histograms
    plt.subplot(3, 3, 7)
    plt.title("Original Histogram")
    plt.hist(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).ravel(), bins=256, range=(0, 256), color='black')
    plt.xlim(0, 256)

    plt.subplot(3, 3, 8)
    plt.title("Sobel Histogram")
    plt.hist(sobel_edges.ravel(), bins=256, range=(0, 256), color='black')
    plt.xlim(0, 256)

    plt.subplot(3, 3, 9)
    plt.title("Canny Histogram")
    plt.hist(canny_edges.ravel(), bins=256, range=(0, 256), color='black')
    plt.xlim(0, 256)

    plt.tight_layout()
    plt.show()

    # Print matrix information
    print("Sobel Edges Matrix:")
    print(sobel_edges)

    print("Canny Edges Matrix:")
    print(canny_edges)

# Path to the image file
image_path = '/content/golden.jpeg'  # Replace with your local image path

# Read the image
image = read_image(image_path)

# Perform Sobel edge detection
sobel_edges = sobel_edge_detection(image)

# Perform Canny edge detection
canny_edges = canny_edge_detection(image)

# Combine Sobel and Canny edges
combined_edges = combine_edges(sobel_edges, canny_edges)

# Superimpose the original image with the combined edge-detected image
superimposed_image = superimpose_images(image, combined_edges)

# Display the images and additional information
display_images(image, sobel_edges, canny_edges, combined_edges, superimposed_image)
