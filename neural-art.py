import numpy as np
import time
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import os
from PIL import Image
import utils


def load_image(image_path, dim = (512, 512)):
    """ Loads and resizes image found at image_path
    
    Args:
        image_path (string): Relative path of image to load

    Returns:
        tf.Tensor: Tensor representation of image
    """

    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_size = max(shape)

    new_shape = tf.cast(dim, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

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
    
    return load_image(img_path)

def load_style_image(filename = '', img_path = ''):
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
    
    return load_image(img_path)

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

def output_to_image(out):
    """ Creates image from output tensor
    
    Args: 
        out (tf.Tensor): Tensor output from model

    Returns:
        PIL.Image: Image represented by tensor
    """

    out = out * 255 # values in tensor are from 0 - 1
    out = np.array(out, dtype = np.uint8)

    if np.ndim(out) > 3:
        out = out[0]

    return Image.fromarray(out) 

def intermediate_layers(layer_names):
    """ Creates a mini-model with desired layer outputs

    Args:
        layer_names (list): list of desired layer output names

    Returns:
        tf.keras.Model: mini-model
    """
    model = tf.keras.applications.VGG19(
            include_top = False, 
            weights = 'imagenet')
    model.trainable = False

    outputs = []
    for name in layer_names:
        outputs.append(model.get_layer(name).output)

    return tf.keras.Model([model.input], outputs)

def bound_values(image):
    return tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 1.0)

def optimize_image(generated_image, target_content, target_style, content_model, style_model, optimizer, impact):
    
    loss = None

    with tf.GradientTape() as tape:
        # Preprocess generated image
        processed = generated_image * 255
        processed = keras.applications.vgg19.preprocess_input(processed)

        # Extract outputs from generated image
        content = content_model(processed)
        style = style_model(processed)

        loss = utils.total_loss(target_content, content, target_style, style, impact)

        grad = tape.gradient(loss, generated_image)

        print(loss)
        
        assert(grad is not None)

        optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(bound_values(generated_image))

    return loss

def calc_eta(n):
    """ Creates a string representation of the ETA of image generation

    Args:
        seconds_left (int): number of seconds left in image generation
    
    Returns:
        string: String representation of ETA

    """
    days = n // (24 * 3600)
    n = n % (24 * 3600)
    hours = n // 3600
    n = n % 3600
    minutes = n // 60
    n = n % 60
    seconds = n

    eta = str(days) + ' days, ' + str(hours) + ' hours, ' + str(minutes) + ' minutes and ' + str(seconds) + ' seconds.'
    return eta

def generate(input_image, style_image, impact, iterations = 100):
    """ Generates resulting image through series of optimizations

    Args:
        input_image (tf.Tensor): Tensor representation of input image
        style_image (tf.Tensor): Tensor representation of style image
        iterations (int): number of optimization iterations

    Returns:
        np array: Generated image
    """

    # Preprocess images
    mod_input = input_image * 255
    mod_input = tf.keras.applications.vgg19.preprocess_input(mod_input)
    
    mod_style = style_image * 255
    mod_style = tf.keras.applications.vgg19.preprocess_input(mod_style)
   
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
            weights = 'imagenet')
    model.trainable = False

    optimizer = tf.optimizers.Adam(learning_rate = 0.02)
    
    # Initialize mini-models
    style_model = intermediate_layers(s_layers)
    content_model = intermediate_layers([c_layer])

    # Store target activations
    target_content = content_model(mod_input)
    target_style = style_model(mod_style)

    # Initialize generated_image
    generated_image = tf.Variable(input_image)

    # Run first step to establish ETA
    print('Step: 1 of ', iterations, ' ETA: Calculating')
    start_time = time.time()
    loss = optimize_image(
            generated_image,
            target_content,
            target_style,
            content_model,
            style_model,
            optimizer,
            impact)
    iter_duration = time.time() - start_time

    # Optimization loop
    for x in range(iterations - 1):
        eta = calc_eta( (iterations - 1 - x) * iter_duration)
        print('Step: ', x + 1, ' of ', iterations, ' ETA: ', eta)

        loss = optimize_image(
                generated_image,
                target_content,
                target_style,
                content_model,
                style_model,
                optimizer,
                impact
                )
        
    return generated_image


def main():
    input_image = load_input_image()
    style_image = load_style_image()
    save_dir = get_save_dir()

    result = generate(input_image, style_image, 'light', 800)
    out = output_to_image(result)
    out.save(save_dir+'_light', format='png')

    result = generate(input_image, style_image, 'medium', 800)
    out = output_to_image(result)
    out.save(save_dir+'_medium', format='png')

    result = generate(input_image, style_image, 'heavy', 800)
    out = output_to_image(result)
    out.save(save_dir+'_heavy', format='png')
    
    print('Output image saved successfully! ')

main()
