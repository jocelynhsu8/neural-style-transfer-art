import numpy as np
from tensorflow.keras.preprocessing import image
import os

def load_input_image():
    filename = ''
    img_path = ''
    
    while True:
        filename = input('Please enter the input image filename and press enter: \n')

        img_path = 'input/' + filename
        if os.path.isfile(img_path):
            break
        
        print('ERROR: Input image not found')
    
    img = image.load_img(img_path)
    input_arr = image.img_to_array(img)
    input_arr = np.array([input_arr])
    
    return input_arr

def load_style_image():
    filename = ''
    img_path = ''
    
    while True:
        filename = input('Please enter the style image filename and press enter: \n')

        img_path = 'style/' + filename
        if os.path.isfile(img_path):
            break
        
        print('ERROR: Style image not found')
    
    img = image.load_img(img_path)
    input_arr = image.img_to_array(img)
    input_arr = np.array([input_arr])
    
    return input_arr



def generate():
    pass

def main():
    input_image = load_input_image()
    style_image = load_style_image()
    print('input: ') 
    print(input_image)
    print('style: ' )
    print(style_image)


main()
