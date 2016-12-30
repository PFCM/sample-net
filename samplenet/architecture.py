"""The actual model."""
import tensorflow as tf
import tensorflow.contrib.slim as slim


def non_overlapping_frames(data, frame_size):
    """Divides an input into non-overlapping frames, returning them
    as a list for ease of use with tensorflow's rnns.

    Will presumably have to engage in some sort of padding of the final
    step, although it might make sense to just discard the final small frame
    depending on the frame size. Assumes all sequences have already been
    padded to equal lengths somehow.

    Args:
        data (tensor): a `[batch_size, time]` sequence of floats.
        frame_size (int): the size of the chunks to divide into.

    Returns:
        list: of `[batch_size, frame_size]` tensors representing the inputs
            at each time step.
    """
    # TODO don't assume frame_size divides the number of samples
    num_split = data.get_shape()[1].value / frame_size
    return tf.split(1, num_split, data)


def autoregressive_mlp(input_frame, shape=None, scope='ar_mlp'):
    """Gets the autoregressive multi-layer perceptron model.

    Args:
        input_frame (tensor): the (possibly embedded and flattened) 2D
            input tensor of floats.
        shape (Optional[list]): list of the numbers of hidden states
            per layer (including the output layer, so it depends how
            you want to construct the distribution). If not specified,
            defaults to [1024, 1024, 256] as per the paper.
        scope (Optional[str or var scope]): variable scope for the
            new variables.

    Returns:
        tensor: `[batch_size, shape[-1]]`, containing logit outputs.
    """
    if not shape:
        shape = [1024, 1024, 256]
    # no batch norm, relus until the last layer
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=None,
                        biases_initializer=tf.constant_initializer(0.0)):
        with tf.variable_scope(scope):
            net = slim.stack(input_frame, slim.fully_connected, shape[:-1])
            net = slim.fully_connected(net, shape[-1], activation_fn=None)
    return net
