import os
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch

mtcnn = MTCNN(
    image_size=160,  # Size to which the face will be resized
    margin=10,  # Add a 10-pixel margin around the detected face
    min_face_size=40,  # Minimum face size to detect
    thresholds=[0.6, 0.7, 0.7],  # Stricter thresholds for face detection
    select_largest=True,  # If multiple faces, select the largest
    post_process=True  # Apply image normalization after cropping
)

# Load FaceNet model for embeddings
facenet = InceptionResnetV1(pretrained='vggface2').eval()

def preprocess(image):
    """
    Preprocess the image for FaceNet using MTCNN.
    Args:
    - image (PIL Image): The image to preprocess.
    Returns:
    - Tensor or None: Cropped and preprocessed face tensor or None if no face is detected.
    """
    # Use MTCNN to detect and crop the face
    face = mtcnn(image)
    return face

def generate_face_encodings(images_folder='Images', encodings_folder='Encodings'):
    """
    Generate face encodings for each person from the images and save all encodings in individual files.
    If an encoding already exists for the person, skip processing that person's folder.

    Args:
    - images_folder (str): Path to the folder containing subfolders of images.
    - encodings_folder (str): Path to save the generated .npy files.
    """
    # Ensure the output folder exists
    os.makedirs(encodings_folder, exist_ok=True)

    for person_folder in os.listdir(images_folder):
        person_path = os.path.join(images_folder, person_folder)

        if os.path.isdir(person_path):  # Check if it's a folder
            person_encoding_folder = os.path.join(encodings_folder, person_folder)

            # Check if the encoding folder already exists for the person
            if os.path.exists(person_encoding_folder) and os.listdir(person_encoding_folder):
                print(f"Encodings already exist for {person_folder}. Skipping.")
                continue  # Skip processing if encodings are already present

            os.makedirs(person_encoding_folder, exist_ok=True)  # Create folder for the person

            print(f"Processing folder: {person_folder}")

            for file_name in os.listdir(person_path):
                file_path = os.path.join(person_path, file_name)

                if file_name.lower().endswith(('jpg', 'jpeg', 'png')):
                    try:
                        img = Image.open(file_path).convert('RGB')  # Ensure RGB format
                        face = preprocess(img)

                        if face is not None:  # If a face is detected and cropped
                            encoding = facenet(face.unsqueeze(0)).detach().numpy()

                            # Save the encoding as a separate .npy file for each image
                            encoding_file = os.path.join(person_encoding_folder, f"{file_name.split('.')[0]}.npy")
                            np.save(encoding_file, encoding)
                            print(f"Processed {file_name}, Encoding shape: {encoding.shape}")
                        else:
                            print(f"No face detected in {file_name}")
                    except Exception as e:
                        print(f"Error processing {file_name}: {e}")

            print(f"Finished processing {person_folder}")

# Run the function
generate_face_encodings('Images', 'Encodings')
