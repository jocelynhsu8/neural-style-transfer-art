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


def main():
    print('hello world')

main()

