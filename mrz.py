from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from PIL import Image

class MRZDetector:
    imagePath = ""
    def __init__(self , imagePath):
       self.imagePath = imagePath

    def getCharachterArray(self):
        mrzImage = self.detectMRZ()
        mrzLines = self.split_mrz(mrzImage)
        mrzCharacters = []
        for line in mrzLines:
            characters_in_line = self.crop_characters(line)
            mrzCharacters.extend(characters_in_line)
        return mrzCharacters



    def detectMRZ(self):
        # Larger structuring element sizes might be needed for larger images
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 20))
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (78, 78))

        # Load the image at its original size or a higher resolution
        image = cv2.imread(self.imagePath)
        # Resize the image maintaining the same aspect ratio but with a higher resolution
        image = imutils.resize(image, height=1488)  # Increased from 400 to 800
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a series of preprocessing steps
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

        # Gradient calculation remains the same
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

        # Morphological operations to close gaps in between letters
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
        thresh = cv2.erode(thresh, None, iterations=4)

        # Remove the edges of the thresholded image
        p = int(image.shape[1] * 0.05)
        thresh[:, 0:p] = 0
        thresh[:, image.shape[1] - p:] = 0

        # Find contours and sort them
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # Initialize ROI in case no MRZ is detected
        roi = None

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            crWidth = w / float(gray.shape[1])
            if ar > 5 and crWidth > 0.75:
                pX = int((x + w) * 0.03)
                pY = int((y + h) * 0.03)
                (x, y) = (x - pX, y - pY)
                (w, h) = (w + (pX * 2), h + (pY * 2))
                roi = image[y:y + h, x:x + w].copy()
                break

        # Ensure an MRZ was detected; otherwise, raise an error
        if roi is None:
            raise ValueError("No MRZ found in image.")
        return roi

    def split_mrz(self , image):

        img_inverted = cv2.bitwise_not(image)

        # Apply thresholding to extract text
        _, img_thresh = cv2.threshold(img_inverted, 200, 255, cv2.THRESH_BINARY)

        # Perform morphological operations (close and dilate)
        kernel = np.ones((10, 100), np.uint8)
        img_closed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        img_dilated = cv2.dilate(img_closed, kernel, iterations=1)

        # Convert the image to single-channel grayscale (CV_8UC1)
        img_dilated_gray = cv2.cvtColor(img_dilated, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(img_dilated_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Pass the converted image to findContours

        # Filter contours to keep only horizontal lines and remove very small areas
        horizontal_lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            if aspect_ratio > 5 and area > 500:
                horizontal_lines.append((x, y, w, h))


        horizontal_lines = sorted(horizontal_lines, key=lambda line: line[1])
        image_lines = []
        for i, (x, y, w, h) in enumerate(horizontal_lines):
            line_img = image[y:y+h, x:x+w]
            image_lines.append(line_img)

        return image_lines




    def crop_characters(self, image, margin=4):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        bounding_boxes.sort(key=lambda x: x[0])
        images = []

        for (x, y, w, h) in bounding_boxes:
            x_start = max(x - margin, 0)
            y_start = max(y - margin, 0)
            x_end = min(x + w + margin, image.shape[1])
            y_end = min(y + h + margin, image.shape[0])
            char_image = image[y_start:y_end, x_start:x_end]
            images.append(char_image)  # Keep as NumPy array
        return images




import os
import cv2
from imutils import paths

# Assuming MRZDetector class is defined elsewhere and imported correctly

def process_images(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all images in the input folder
    image_paths = list(paths.list_images(input_folder))
    for image_path in image_paths:
        # Initialize the MRZDetector with the current image
        mrz_detector = MRZDetector(image_path)
        try:
            # Process the image to detect the MRZ
            processed_image = mrz_detector.detectMRZ()
            # Construct the output image path
            output_image_path = os.path.join(output_folder, os.path.basename(image_path))
            # Save the processed image
            cv2.imwrite(output_image_path, processed_image)
            print(f"Processed and saved: {output_image_path}")
        except Exception as e:
            print(f"Failed to process {image_path}: {str(e)}")


import os
import cv2
from imutils import paths

def process_first_image(input_folder, output_folder):
    # Ensure the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of image paths in the folder
    image_paths = list(paths.list_images(input_folder))
    
    if not image_paths:
        raise ValueError("No images found in the specified folder.")

    # Select the first image
    for image_path_index, image_path in enumerate(image_paths):
    
    # Initialize the MRZDetector with the first image
        mrz_detector = MRZDetector(image_path)
        
        try:
            # Detect the MRZ area and split it into MRZ lines
            mrz_image = mrz_detector.detectMRZ()
            mrz_lines = mrz_detector.split_mrz(mrz_image)
            
            # Save each MRZ line image
            for line_index, line in enumerate(mrz_lines):
                line_image_path = os.path.join(output_folder, f"{image_path_index}_mrz_line_{line_index + 1}.png")
            #  cv2.imwrite(line_image_path, line)
                print(f"Saved MRZ line image: {line_image_path}")
                
                # Get characters from each line and save them
                characters = mrz_detector.crop_characters(line)
                for char_index, char in enumerate(characters):
                    char_image_path = os.path.join(output_folder, f"{image_path_index}_mrz_line_{line_index + 1}_char_{char_index + 1}.png")
                    cv2.imwrite(char_image_path, char)
                    print(f"Saved character image: {char_image_path}")

        except Exception as e:
            print(f"Failed to process : {str(e)}")



# Example usage
#input_folder = 'srb_passport'
#output_folder = 'srb_characters'
#process_first_image(input_folder, output_folder)


# Example usage
#input_folder = 'srb_passport'
#output_folder = 'srb_output'
#process_images(input_folder, output_folder)

