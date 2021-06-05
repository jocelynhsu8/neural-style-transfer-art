import numpy as np

def content_loss(content_img, gen_img):
    """ Calculate mean squared error between content image and generated image.

    Args:
        contentimg (np array): original input image
        gen_img (np array): [description]

    Returns:
        int: Calculated mean squared error
    """
    if content_img.shape != gen_img.shape:
        print('Images have different dimensions')
        exit()
    sum = 0
    for i in range(1, content_img.shape[1] + 1):
        for j in range(1, content_img.shape[2] + 1):
            for k in range(1, 4):
                sum += (content_img[1][i][j][k] - gen_img[1][i][j][k]) ** 2
            
    sum /= content_img.shape[1] * content_img.shape[2] * 3
    return sum

def main():
    print('hello world')

main()

