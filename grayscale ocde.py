import numpy as np
import argparse
import os
import cv2
import pyperclip

# Set the dimensions of the photo
width, height = 700, 500

# Open the default camera (usually camera index 0)
cap = cv2.VideoCapture(0)

# Set the dimensions of the capture
cap.set(3, width)  # Width
cap.set(4, height)  # Height

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create a folder to save photos if it doesn't exist
photo_folder = "photos"
os.makedirs(photo_folder, exist_ok=True)

# Capture a photo when 'o' key is pressed
ret, frame = cap.read()

# Check if the frame was captured successfully
if not ret:
    print("Error: Could not capture photo.")
    exit()

# Convert the photo to grayscale
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Display the original and grayscale images
cv2.imshow("Grayscale", gray_frame)

# Save the grayscale image with a numbered filename
photo_number = 1
filename = os.path.join(photo_folder, f"photo_{photo_number:06d}.jpg")
cv2.imwrite(filename, gray_frame)

# Copy the path of the saved photo to the clipboard with the command line argument
command_line_argument = f'python gray.py -i "{filename}"'
pyperclip.copy(command_line_argument)
image_path = filename  # Corrected assignment

print(f"Photo saved: {filename}")
print("Command line argument copied to clipboard")

# Release the camera and close windows
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

# Paths to load the model
DIR = r"C:\Users\rahul\Desktop\Project"
PROTOTXT = os.path.join(DIR, r"C:\Users\rahul\Desktop\Project\colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"C:\Users\rahul\Desktop\Project\pts_in_hull.npy")
MODEL = os.path.join(DIR, r"C:\Users\rahul\Desktop\Project\colorization_release_v2.caffemodel")

# Argparser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to input black and white image")
args = vars(ap.parse_args())

# Load the Model
print("Load model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load the input image
image = cv2.imread(args["image"])

# Debug: Print the image shape
print(f"Image shape: {image.shape}")

scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

# Debug: Print the shape of important variables
print(f"Resized shape: {resized.shape}")
print(f"L shape: {L.shape}")

# Colorizing the image
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

colorized = (255 * colorized).astype("uint8")

cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)
