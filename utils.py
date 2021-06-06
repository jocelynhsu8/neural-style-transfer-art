import numpy as np

def content_loss(content_img, gen_img):
    """ Calculate mean squared error between activations of content image and generated image.

    Args:
        contentimg (np array): content image
        gen_img (np array): generated image

    Returns:
        int: Computed mean squared error of activation
    """
    if content_img.shape != gen_img.shape:
        print('Images have different dimensions')
        exit()
    error = 0
    for i in range(1, content_img.shape[1] + 1):
        for j in range(1, content_img.shape[2] + 1):
            for k in range(1, content_img.shape[3] + 1):
                error += (content_img[1][i][j][k] - gen_img[1][i][j][k]) ** 2
            
    error /= content_img.shape[1] * content_img.shape[2] * content_img.shape[3]
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

def style_loss_ind(style_gram_mat, gen_gram_mat, weight): 
    """ Calculate mean squared error of individual filter

    Args:
        style_img (np array): Gram matrix of style image
        gen_img (np array): Gram matrix of generated image
        weight (double): weight of layer

    Returns:
        [int]: weighted mean squared error of filter
    """
    if style_gram_mat.shape != gen_gram_mat.shape:
        print('Images have different dimensions')
        exit()
    error = 0
    for i in range(1, style_gram_mat.shape[1] + 1):
        for j in range(1, style_gram_mat.shape[2] + 1):
            for k in range(1, style_gram_mat.shape[3] + 1):
                error += (style_gram_mat[1][i][j][k] - gen_gram_mat[1][i][j][k]) ** 2
            
    error /= style_gram_mat.shape[1] * style_gram_mat.shape[2] * style_gram_mat.shape[3]
    return error * weight

def style_loss_overall(style_img, gen_img, num_layers = 5, weight = []):
    """ Calculate overall weighted mean squared error of all layers

    Args:
        style_img (np array): Style image
        gen_img (np array): Generated image
        num_layers (int, optional): Number of layers. Defaults to 5.
        weight (list, optional): List of weights for each layer. Defaults to [].

    Returns:
        [int]: Overall weighted mean squared error of style image
    """
    if len(weight) == 0: # equal weight of each layer if weight not specified
        for i in range(1,num_layers + 1):
            weight.append(1 / num_layers)
    error = 0
    style_gram_mat = calc_gram(style_img)
    gen_gram_mat = calc_gram(gen_img)
    for i in range(1, num_layers + 1):
        error += style_loss_ind(style_gram_mat, gen_gram_mat, weight[i - 1]) # create output image?
    return error
    


def main():
    print('hello world')

main()

