# Projects
# Grayscale Image Capture & Colorization

This project captures an image using a webcam, converts it to grayscale, and saves it. Additionally, it uses a deep learning model to recolorize the grayscale image.

## Features
- Captures an image from the webcam.
- Converts the image to grayscale and saves it.
- Copies the image path as a command-line argument to the clipboard.
- Uses a pre-trained deep learning model to colorize grayscale images.
- Displays both grayscale and recolorized images.

## Requirements
Ensure you have the following dependencies installed before running the script:

```sh
pip install numpy opencv-python pyperclip
```

## Usage
### 1. Run the Script
```sh
python grayscale_ocde.py
```

### 2. Capture and Save the Image
- The script opens the webcam and captures an image.
- The image is converted to grayscale and saved in the `photos/` directory.
- The command to process the image (`gray.py -i "path_to_image"`) is copied to the clipboard.

### 3. Recolorize the Image
- The script loads a **pre-trained Caffe model** for colorization.
- Processes the grayscale image and generates a colorized output.
- Displays both the original grayscale and the recolorized images.

## File Structure
```
├── grayscale_ocde.py  # Main script
├── photos/            # Directory where images are saved
├── model/             # Pre-trained deep learning model files
├── README.md          # Project documentation
```

## Model Files
To enable colorization, ensure you have the following files in the `model/` directory:
- `colorization_deploy_v2.prototxt`
- `colorization_release_v2.caffemodel`
- `pts_in_hull.npy`

## Notes
- Make sure your webcam is connected and accessible.
- The script will exit if the camera cannot be accessed.

## License
This project is open-source and available for personal and educational use.

---
Feel free to contribute or modify the code as needed!

