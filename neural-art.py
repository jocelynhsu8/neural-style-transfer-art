import numpy as np
from tensorflow.keras.preprocessing import image
import os
from PIL import Image

# TODO: Will need to resize image based on network
def load_input_image(filename = '', img_path = ''):
    """ Load input image based on user-specified filename.

    Returns:
        np array: Input (content) image
    """
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

# TODO: Will need to resize image based on network
def load_style_image(filename = '', img_path = ''):
    """ Load style image based on user-specified filename.

    Returns:
        np array: Style image
    """
    
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

def get_save_dir():
    """ Prompt user to receive valid save filename

    Returns:
        string: Directory to valid save location
    """
    
    while True:
        filename = input('Please enter the filename to save the resulting image: \n')

        save_dir = 'output/' + filename
        if not os.path.isfile(save_dir):
            break
        
        print('ERROR: File with given filename already exists!')
    return save_dir

# TODO:
def generate(input_image, style_image, iterations = 50):
    return input_image

def main():
    input_image = load_input_image()
    style_image = load_style_image()
    save_dir = get_save_dir()

    result = generate(input_image, style_image)
    result = np.clip(result[0], 0, 255).astype('uint8')
    result_image = Image.fromarray(result)
    result_image.save(save_dir)
    
    print('Output image saved successfully! ')

main()
