from tensorflow.keras.models import load_model
from mrz import MRZDetector  # Importing the MRZ class from mrz.py
from PIL import Image
import numpy as np
import os
import shutil
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pytesseract
import re
class FraudDetectionModel:
    def __init__(self):
      
        self.model = load_model('static/custom7.h5')  # Load the pre-trained model from the specified path
        self.datagen = ImageDataGenerator(rescale=1./255) 
        self.char_folder = 'character_images'
        
         # Ensure this matches training
    def save_characters(self, char_images):
        # Path to the test directory
        test_dir = os.path.join(self.char_folder, "test")
        
        # Delete the test directory if it already exists
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        # Create directory if it does not exist
        os.makedirs(test_dir, exist_ok=True)
        
        # Save each character image to the folder
        for i, img in enumerate(char_images):
            img_path = os.path.join(test_dir, f'char_{i}.png')
            img_pil = Image.fromarray(img.astype('uint8'), 'RGB')
            img_pil.save(img_path)
            img = cv2.imread(img_path)
        
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            # Convert the grayscale image back to three channels
            bw_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        
            # Overwrite the original image with the grayscale image
            cv2.imwrite(img_path, bw_image)
            
    
    def detect_fraud2(self , image_path):
        mrz_processor = MRZDetector(image_path)
        char_images = mrz_processor.getCharachterArray()
        
        # Save the extracted characters to files
        self.save_characters(char_images)
        
        # Setup the ImageDataGenerator for inference
        datagen = ImageDataGenerator(rescale=1./255)
        generator = datagen.flow_from_directory(
            self.char_folder,
            target_size=(32, 32),
            color_mode='rgb',
            class_mode='binary',
            batch_size=1,
            shuffle=False
        )
        
        # Predict using the model and display each image
        y_true = []
        y_pred = []
        print(len(generator))
        imagecount = 0
        # Retrieve images and labels batch by batch
        for j in range(len(generator)):
            imgs, labels = generator.next()
            for i in range(len(imgs)):
                imagecount+=1
                #print(str(imagecount) + " out of "+str(len(generator) * 1))
                img = np.expand_dims(imgs[i], axis=0)  # Expand dims to make it (1, 32, 32, 3)
                prediction = self.model.predict(img)
                predicted_class = np.round(prediction).astype(int)
                true_class = labels[i]
                print(str(prediction) )
                print(generator.filenames[j])


                # Append to lists
                y_pred.append(predicted_class[0])
                y_true.append(int(true_class))
                if (predicted_class == 0):
                    return 0 , ""
        text = self.get_ocr(self.char_folder+"/test")        
        return 1 , text

    def detect_fraud(self):
        mrz_processor = MRZDetector(self.image_path)
        char_images = mrz_processor.getCharachterArray()

        print(f"Total characters extracted: {len(char_images)}")
        
        for img in char_images:
            # Convert image to PIL, resize, and convert back to array
            img_pil = Image.fromarray(img.astype('uint8'), 'RGB')
            resized_img = img_pil.resize((32, 32))
            img_array = np.array(resized_img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # Use ImageDataGenerator for consistent preprocessing
            img_processed = self.datagen.flow(img_array, batch_size=1)
            
            # Display the image being predicted
            plt.imshow(img_processed[0][0])  # Correctly display the first image in the batch
            plt.title("Image being classified")
            plt.show()
            
            # Predict the class of each image using the loaded model
            prediction = self.model.predict(img_processed)
            predicted_class = (prediction > 0.5).astype("int32")[0][0]
            if (predicted_class == 0):
                return 0

        return 1
        
    @staticmethod
    def natural_sort_key(s):
     return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    def get_ocr(self, input_folder):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Retrieve and sort files naturally
        files = os.listdir(input_folder)
        files = sorted(files, key=FraudDetectionModel.natural_sort_key)
        
        textocr = ""
        for file_name in files:
            image_path = os.path.join(input_folder, file_name)
            image = Image.open(image_path)
            
            # Use pytesseract to extract text
            text = pytesseract.image_to_string(image, config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<>')
            textocr += text

        return textocr

# Example usage:
# Assuming you have an image at "path/to/image.jpg"
#detector = FraudDetectionModel("00.jpg")
#result = detector.detect_fraud2()
#print("Fraud Detection Result:", result)