import cv2
import yaml
import numpy as np
from pathlib import Path

# Function to compute keypoints and descriptors using SIFT
def extract_features(image, feature_detector="SIFT", nfeatures=500):
    if feature_detector == "SIFT":
        sift = cv2.SIFT_create(nfeatures)  # Using SIFT
        keypoints, descriptors = sift.detectAndCompute(image, None)
    else:
        raise ValueError("Unsupported feature detector. Use 'SIFT'.")
    return keypoints, descriptors

# Function to compute overlap score between two images using feature matching
def compute_overlap(image1, image2, detector="SIFT", match_threshold=0.75, nfeatures=500):
    # Extract features
    kp1, des1 = extract_features(image1, feature_detector=detector, nfeatures=nfeatures)
    kp2, des2 = extract_features(image2, feature_detector=detector, nfeatures=nfeatures)
    
    # Check if descriptors are valid
    if des1 is None or des2 is None:
        print("One of the descriptors is empty. Returning zero overlap.")
        return 0  # No overlap
    
    # Match features using BFMatcher with L2 norm for SIFT
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    # Ensure that descriptors are not empty and have the same dimension (number of columns)
    if des1.shape[0] == 0 or des2.shape[0] == 0:
        print("No keypoints detected in one or both images.")
        return 0  # No overlap
    
    # Perform knnMatch (k=2 for Lowe's ratio test)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < match_threshold * n.distance:
            good_matches.append(m)
    
    # Calculate overlap score as the ratio of good matches to the minimum keypoints count
    overlap_score = len(good_matches) / min(len(kp1), len(kp2)) if min(len(kp1), len(kp2)) > 0 else 0
    return overlap_score

# Image sampling based on overlap calculation
def sample_images(folder_path, overlap_threshold=0.2, detector="SIFT", nfeatures=500):
    # Get all .JPG files in the folder
    image_files = sorted(Path(folder_path).glob("*.JPG"))
    if not image_files:
        raise ValueError("No .JPG images found in the specified folder.")
    
    # Read the first image and initialize the sampled set
    sampled_images = [cv2.imdecode(np.fromfile(str(image_files[0]), dtype=np.uint8), -1)]
    sampled_filenames = [image_files[0].name]
    
    for i, image_path in enumerate(image_files[1:], start=2):
        current_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        overlap = compute_overlap(current_image, sampled_images[-1], detector=detector, nfeatures=nfeatures)
        if overlap <= overlap_threshold:
            sampled_images.append(current_image)
            sampled_filenames.append(image_path.name)
            print(f"Image {image_path.name} added to the sampled set.")
        else:
            print(f"Image {image_path.name} skipped due to high overlap.")
    
    return sampled_filenames

# Main execution
if __name__ == "__main__":
    with open('config/config.yml', 'r', encoding="utf-8") as file:
            user_config = yaml.safe_load(file)
            
    sampled_images = sample_images(folder_path=user_config["image_folder"], 
                                   
                                   overlap_threshold=user_config["similarity_threshold"], 
                                   
                                   detector="SIFT",
                                   
                                   nfeatures=user_config["nfeatures"])
    
    print("Sampled images:", sampled_images)
