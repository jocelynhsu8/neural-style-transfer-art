import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import os
from PIL import Image
import utils

def load_input_image(filename = '', img_path = '', dim = (400, 400)):
    """ Load input image based on user-specified filename. Images are resized to 400 x 400.

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
    data = image.img_to_array(img)
    input_image = np.array([data])
    input_image = image.smart_resize(input_image[0], dim, interpolation = 'bilinear') 
    return input_image


def load_style_image(filename = '', img_path = '', dim = (400, 400)):
    """ Load style image based on user-specified filename. Images are resized to 640 x 480.

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
    data = image.img_to_array(img)
    style_image = np.array([data])
    style_image = image.smart_resize(style_image[0], dim, interpolation = 'bilinear')
    return style_image

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

def intermediate_layers(layer_names):
    """ Creates a mini-model with desired layer outputs

    Args:
        layer_names (list): list of desired layer output names

    Returns:
        tf.keras.Model: mini-model
    """
    model = tf.keras.applications.VGG19(
            include_top = False, 
            weights = 'imagenet', 
            input_shape = (400, 400, 3))
    model.trainable = False

    outputs = []
    for name in layer_names:
        outputs.append(model.get_layer(name).output)

    return tf.keras.Model([model.input], outputs)

@tf.function()
def training_step(image, optimizer, total_loss):
    """ Performs one step of gradient descent to optimize image
    
    Args:
        image (tf.Variable): generated image to optimize
        total_loss (?): total loss (style and content)
        optimizer (tf.optimizers): TF Optimizer (Adam)
    """
    tape = tf.GradientTape()
    grad = tape.gradient(total_loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(bound_values(image))

def bound_values(image):
    return tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 1.0)

# TODO: FINISH IMPLEMENTATION
def generate(input_image, style_image, iterations = 200):
    """ Generates resulting image through series of optimizations

    Args:
        input_image (np array): content image
        style_imgae (np array): style image
        iterations (int): number of optimization iterations

    Returns:
        np array: Generated image
    """
    # Expand dimensions to account for batch_size when sending to model
    mod_input = np.expand_dims(input_image, axis = 0)
    mod_style = np.expand_dims(style_image, axis = 0)

    # Let generated_image be replica of input to begin with (May later change to noise)
    mod_gen = mod_input
   
    # Store content & style layers
    c_layer = 'block4_conv2'
    s_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
    
    # Initialize VGG19 model
    model = tf.keras.applications.VGG19(
            include_top = False, 
            weights = 'imagenet', 
            input_shape = (400, 400, 3))
    optimizer = tf.optimizers.Adam(learning_rate=0.02)
    
    # Initialize mini-models
    style_model = intermediate_layers(s_layers)
    content_model = intermediate_layers([c_layer])
    
    # Optimization loop
    for x in range(iterations):
        print('Step: ', x)
        # Compute  loss
        c_activ = content_model(mod_input)
        g_c_activ = content_model(mod_gen)

        s_activ = style_model(mod_style)
        g_s_activ = style_model(mod_gen)

        total_loss = utils.total_loss(c_activ, g_c_activ, s_activ, g_s_activ)

        # Update generated image
        training_step(mod_gen, optimizer, total_loss)
        
        # Print every 10 iterations to track progress
        if x % 10 == 0:
            print('Iteration #: ', x)
            print('Total loss: ', total_loss)

    return mod_gen


def main():
    input_image = load_input_image()
    style_image = load_style_image()
    save_dir = get_save_dir()

    result = generate(input_image, style_image)

    #result = np.clip(result, 0, 255).astype('uint8')
    #result_image = Image.fromarray(result)
    #result_image.save(save_dir)
    
    print('Output image saved successfully! ')

main()
