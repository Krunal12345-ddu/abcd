import cv2
import numpy as np

def calculate_ssim(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate Structural Similarity Index (SSI)
    ssim_index, _ = cv2.compareSSIM(gray1, gray2, full=True)

    return ssim_index

# Example usage
if __name__ == "__main__":
    # Load two images for comparison
    image1 = cv2.imread("open_cv_frame_0.png")
    image2 = cv2.imread("open_cv_frame_2.png")

    # Check if images are loaded successfully
    if image1 is None or image2 is None:
        print("Error: Could not load images.")
    else:
        # Calculate SSIM
        similarity_index = calculate_ssim(image1, image2)

        # Print the similarity index
        print(f"Similarity Index: {similarity_index}")

        # Define a threshold for similarity
        threshold = 0.9  # Adjust as needed

        # Check if images are similar based on the threshold
        if similarity_index > threshold:
            print("Images are similar.")
        else:
            print("Images are not similar.")
