import tensorflow as tf

def content_loss(content_img, gen_img):
    """ Calculate mean squared error between activations of content image and generated image.

    Args:
        content_ing (tf.Tensor): Activation tensor for content image
        gen_img (tf.Tensor): Activation tensor for generated image

    Returns:
        int: Computed mean squared error of activation

    """
    
    error = tf.add_n([tf.reduce_mean((content_img - gen_img) ** 2)])
    return error

def calc_gram(filter):
    """ Calculate gram matrix to find correlations of feature maps in single layer.

    Args:
        filter (np array): Individual layer

    Returns:
        [np array]: Gram matrix of layer
    """
    return tf.linalg.matmul(filter, filter, transpose_b = True)
    # filter_t = np.transpose(filter)
    # return np.matmul(filter, filter_t)

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

    error = tf.add_n([tf.reduce_mean((style_gram_mat - gen_gram_mat) ** 2)])    
    return error * weight

def style_loss_overall(style_img, gen_img, weight = []):
    """ Calculate overall weighted mean squared error of all layers

    Args:
        style_img (list of tf.Tensor): List of activation tensors for style image
        gen_img (list of tf.Tensor): List of activation tensors for generated image
        weight (list, optional): List of weights for each layer. weight[i] is weight of layer i - 1. Defaults to [].

    Returns:
        [double]: Overall weighted mean squared error of style image
    """

    num_layers = len(style_img) 

    if len(weight) == 0: # descending geometric progression each layer if weight not specified
        cur = 0.5
        sum_weight = 0.5 * (1 - 0.5 ** num_layers) / 0.5 # normalize total sum of weights to 1
        for i in range(0,num_layers):
            weight.append(cur / sum_weight)
            cur *= 0.5
    
    error = 0
    
    for i in range(0, num_layers):
        error += style_loss_ind(style_img[i], gen_img[i], weight[i])
    error /= num_layers

    assert(not tf.math.is_inf(error))
    
    return error
    
def total_loss(content_img, content_gen_img, style_img_list, style_gen_img_list, alpha = 0.5, beta = 0.5, weight = []):
    """ Calculate total loss with weighted content and style mean squared errors

    Args:
        content_img (tf.Tensor): Activation tensor for content image
        content_gen_img (tf.Tensor): Activation tensor for content layer on generated image
        style_img_list (list of tf.Tensor): List of activation tensors for style image
        style_gen_img_list (list of tf.Tensor): List of activation tensors for style layers on generated image
        alpha (double): Weight for content error. Defaults to 0.5.
        beta (double): Weight for style error. Defaults to 0.5.
        weight (list, optional): List of weights for style image filters. Defaults to [].

    Returns:
        [double]: Overall error
    """
    return alpha * content_loss(content_img, content_gen_img) + beta * style_loss_overall(style_img_list, style_gen_img_list, weight)



