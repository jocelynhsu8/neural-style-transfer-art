import numpy as np
import tensorflow as tf

def content_loss(content_img, gen_img):
    """ Calculate mean squared error between activations of content image and generated image.

    Args:
        contentimg (np array): content image
        gen_img (np array): generated image

    Returns:
        int: Computed mean squared error of activation

    if content_img.shape != gen_img.shape:
        print('Images have different dimensions')
        exit()
    error = 0
    for i in range(0, content_img.shape[1]):
        print('i : ', i, ' out of: ', content_img.shape[1])
        for j in range(0, content_img.shape[2]):
            for k in range(0, content_img.shape[3]):
                error += (content_img[0][i][j][k] - gen_img[0][i][j][k]) ** 2
            
    error /= content_img.shape[1] * content_img.shape[2] * content_img.shape[3]
    """
    
    error = tf.add_n([tf.reduce_mean((content_img - gen_img) ** 2)])
    print('Computed content_loss: ', error)
    return error

def calc_gram(filter):
    """ Calculate gram matrix to find correlations of feature maps in single layer.

    Args:
        filter (np array): Individual layer

    Returns:
        [np array]: Gram matrix of layer
    """
    filter_t = np.transpose(filter)
    return np.matmul(filter, filter_t)

def style_loss_ind(style_img, gen_img, weight): 
    """ Calculate mean squared error of individual filter.

    Args:
        style_img (np array): Gram matrix of style image
        gen_img (np array): Gram matrix of generated image
        weight (double): Weight of layer

    Returns:
        [double]: Weighted mean squared error of filter
    """
    style_gram_mat = calc_gram(style_img)
    gen_gram_mat = calc_gram(gen_img)
    if style_gram_mat.shape != gen_gram_mat.shape:
        print('Images have different dimensions')
        exit()
    error = 0
    for i in range(0, style_gram_mat.shape[1]):
        for j in range(0, style_gram_mat.shape[2]):
            for k in range(0, style_gram_mat.shape[3]):
                error += (style_gram_mat[1][i][j][k] - gen_gram_mat[1][i][j][k]) ** 2
            
    error /= style_gram_mat.shape[1] * style_gram_mat.shape[2] * style_gram_mat.shape[3]
    return error * weight

def style_loss_overall(style_img, gen_img, weight = []):
    """ Calculate overall weighted mean squared error of all layers

    Args:
        style_img (list of np array): List of style images
        gen_img (list of np array): List of generated images
        weight (list, optional): List of weights for each layer. weight[i] is weight of layer i - 1. Defaults to [].

    Returns:
        [double]: Overall weighted mean squared error of style image
    """
    if len(weight) == 0: # equal weight of each layer if weight not specified
        num_layers = len(style_img)
        for i in range(0,num_layers):
            weight.append(1 / num_layers)
    error = 0
    for i in range(0, num_layers):
        style_gram_mat = calc_gram(style_img[i])
        gen_gram_mat = calc_gram(gen_img[i])
        error += style_loss_ind(style_gram_mat, gen_gram_mat, weight[i])
    return error
    
def total_loss(content_img, content_gen_img, style_img_list, style_gen_img_list, alpha = 0.5, beta = 0.5, weight = []):
    """ Calculate total loss with weighted content and style mean squared errors

    Args:
        content_img (np array): Content image
        content_gen_img (np array): Generated image for content error calculation
        style_img_list (list of np arrays): List of style image filters
        style_gen_img_list (list of np arrays): List of generated image filters for style error calculation
        alpha (double): Weight for content error
        beta (double): Weight for style error
        weight (list, optional): List of weights for style image filters. Defaults to [].

    Returns:
        [double]: Overall error
    """
    return alpha * content_loss(content_img, content_gen_img) + beta * style_loss_overall(style_img_list, style_gen_img_list, weight)



