"""The actual model."""
import itertools

import tensorflow as tf
import tensorflow.contrib.slim as slim


def temporal_upsample(sequence, rate, method='perforated', ksize=None):
    """Upsamples a sequence into a longer sequence, provides a few methods.

    Args:
        sequence (tensor): `[time, batch_size, num_features]` tensor to stretch
            out. This can be a length `time` list of `[batch_size, features]`
            tensors as well, which is what you tend to get from tf's rnn
            functions.
        rate (int): the factor by which to upsample
        method (Optional[str]): the method by which to perform the upsampling.
            Available methods are:
            - 'perforated': as per the paper, equivalent to inserting zeros and
              then convolving
            - 'bilinear': just bilinearly interpolate, no learned params
            - 'bicubic': bicubic interpolation for smoother vibes
            - 'nearest_neighbor': just copy the nearest ones
            - add '_conv' to the end of any of the non-learned ones to
              optionally apply a learned convolution to them after the
              resizing step.
        ksize (Optional[list of int]): the size of the kernel for any
            convolutions. The first dimension is time and the second is
            features. The default is `[3, 1]` which will treat each feature
            channel independently when resampling. The convolutions acts
            across features produced by RNNs, so there is unlikely to be any
            point having the second number greater than `1`, as there is no
            reason to assume that nearby features have any relation. Could
            be interesting to set it to use the lot, but probably slow.
            It's also important to note that you will almost definitely want
            the size of the convolution to be at least rate+1 so that it
            actually sees a couple of real samples.
    Returns:
        tensor: `[time*rate, batch_size, num_features]` expanded tensor or list
            of length `time*rate` containing `[batch_size, num_features]`
            tensors depending on the provided input format.

    Raises:
        ValueError: if `method` is uninterpretable.
    """
    if ksize is None:
        ksize = [3, 1]

    if isinstance(sequence, list):
        sequence = tf.stack(sequence)
        return_list = True
    else:
        return_list = False

    # turn it into a [batch, time, features, 1] image
    sequence_img = tf.transpose(sequence, [1, 0, 2])
    sequence_img = tf.expand_dims(sequence_img, -1)
    im_shape = sequence_img.get_shape().as_list()  # handy to keep around
    new_size = im_shape[1:3]
    new_size[0] *= rate
    if method == 'perforated':
        # this is just a classic conv_transpose
        # docs for slim suggest having different strides won't work, but this
        # appears to be false
        sequence_img = slim.conv2d_transpose(
            sequence_img,
            1,  # one output channel
            ksize,
            [rate, 1],  # stride, only expanding one dimension
            padding='SAME',  # I think
            activation_fn=None,
            normalizer_fn=None,
            normalizer_params=None)
    elif 'bilinear' in method:
        sequence_img = tf.image.resize_bilinear(sequence_img, new_size,
                                                align_corners=True)
    elif 'bicubic' in method:
        sequence_img = tf.image.resize_bicubic(sequence_img, new_size,
                                               align_corners=True)
    elif 'nearest_neighbor' in method:
        sequence_img = tf.image.resize_nearest_neighbor(sequence_img, new_size,
                                                        align_corners=True)
    else:
        raise ValueError('unknown interpolation: {}'.format(method))

    if 'conv' in method:
        # perforated does it all in one step, so no point convolving twice
        # but it will be ignored
        sequence_img = slim.conv2d(sequence_img,
                                   1,  # 1 input channel -> 1 output channel
                                   ksize,  # size of the window
                                   [1, 1],  # stride, needs to be fixed
                                   padding='SAME',  # keep the shape
                                   activation_fn=None,  # linear
                                   normalizer_fn=None,  # no batch norm here
                                   biases_initializer=tf.constant_initializer(
                                       0.0),  # otherwise slim doesn't use them
                                   scope='smooth')
    # now get back to the correct format
    new_sequence = tf.squeeze(sequence_img, -1)
    new_sequence = tf.transpose(new_sequence, [1, 0, 2])  # make time major

    if return_list:
        new_sequence = tf.unstack(new_sequence)

    return new_sequence


def non_overlapping_frames(data, frame_size, time_major=False):
    """Divides an input into non-overlapping frames, returning them
    as a list for ease of use with tensorflow's rnns.

    Will presumably have to engage in some sort of padding of the final
    step, although it might make sense to just discard the final small frame
    depending on the frame size. Assumes all sequences have already been
    padded to equal lengths somehow.

    Args:
        data (tensor): depending on the `time_major` flag, either a
            `[time, batch]` or `[batch, time]` sequence of floats.
        frame_size (int): the size of the chunks to divide into.
        time_major (Optional[bool]): whether the first axis of `data` is time
            (if True) or batch (if False). Default is False. This does not
            change the way the output is returned.

    Returns:
        list: of `[batch_size, frame_size]` tensors representing the inputs
            at each time step.
    """
    # TODO don't assume frame_size divides the number of samples
    if time_major:
        data = tf.transpose(data)
    num_split = data.get_shape()[1].value / frame_size
    return tf.split(1, num_split, data)


def autoregressive_mlp(input_frame, shape=None, scope='ar_mlp'):
    """Gets the autoregressive multi-layer perceptron model.

    If we do exactly what the paper says, the lowest layer of this thing will
    be monstrous ([frame_size*embedding_size, 1024]), which is a bit absurd
    given the clear temporal dependencies we should expect. It might be more
    reasonable to apply convolutions along the temporal axis (in which case
    it may prove sensible to forego the embeddings in favour of more filters)

    Args:
        input_frame (tensor): the (possibly embedded and flattened) 2D
            input tensor of floats. If not 2D, then we flatten it and reshape
            the output appropriately
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
    if len(input_frame.get_shape()) != 2:
        original_shape = input_frame.get_shape().as_list()
        # flatten out the batch of windows
        input_frame = tf.reshape(input_frame, [original_shape[0], -1])
    else:
        original_shape = None
    # no batch norm, relus until the last layer
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=None,
                        biases_initializer=tf.constant_initializer(0.0)):
        with tf.variable_scope(scope):
            net = slim.stack(input_frame, slim.fully_connected, shape[:-1])
            net = slim.fully_connected(net, shape[-1], activation_fn=None)
    # if original_shape:
    #     original_shape[-1] = shape[-1]
    #     net = tf.reshape(net, original_shape, name='mlp_out')
    # else:
            net = tf.identity(net, name='mlp_out')
    return net


def get_weightnormed_variable(shape, scope=None, **kwargs):
    """gets a variable with weight normalisation applied. At this stage just
    matrices, and weightnorm is applied on the _columns_ only.

    Args:
        shape (list): the shape of the variable, needs to be length 2.
        **kwargs: any other args passed into get_variable when getting the
            variables required (not the gains).
    """
    if not scope:
        scope = 'WN_Variable'
    if len(shape) != 2:
        raise ValueError(
            'Can only weightnorm matrices, got shape: {}'.format(shape))
    with tf.variable_scope(scope):
        weights = tf.get_variable(scope + '_weights', shape=shape, **kwargs)
        # ones in the shape make sure they broadcast appropriately
        gains = tf.get_variable(scope + '_gain', shape=[shape[0], 1],
                                initializer=tf.constant_initializer(1.0))
        offsets = tf.get_variable(scope + '_offset', shape=[shape[0], 1],
                                  initializer=tf.constant_initializer(0.0))

        normed = weights * tf.rsqrt(tf.reduce_sum(tf.square(weights),
                                                  reduction_indices=[1],
                                                  keep_dims=True))
        return (normed * gains) + offsets


def linear_projection(inputs, output_size=None, rate=1, weightnorm=True,
                      scope=None):
    """Performs linear projections on a sequence of data. Optionally cycles
    through a number of different projections, so can be used for upsampling
    in the manner talked about in the paper.

    Args:
        inputs (list): list of inputs, each should be
            `[batch_size, num_features]`
        output_size (Optional[int]): the number of features we are projecting
            to.
        rate (Optional[int]): the number of different projections we apply
            to each element of the input. Default just applies the same
            projection to each input. If > 1 the result will be a so-called
            'perforated' upsampling of the input.
        weightnorm (Optional[bool]): whether to use weight normalisation on the
            matrices. Defaults to True.
        scope (Optional): variable scope or name of one under which to add ops
            and get variables.

    Returns:
        list: len(inputs) * rate length list of `[batch_size, output_size]`
            tensors.
    """
    with tf.variable_scope(scope or 'projection'):
        # it is pretty much always faster to do fewer big matmuls
        # so the first thing we'll do is stick inputs together into one big
        # matrix
        batch_size, input_size = inputs[0].get_shape().as_list()
        inputs = tf.concat(0, inputs)
        if not output_size:
            output_size = input_size
        # now we'll project it `rate` times
        projections = []
        for proj in range(rate):
            with tf.variable_scope('projection_{}'.format(proj)):
                if weightnorm:
                    weights = get_weightnormed_variable(
                        [input_size, output_size], 'proj_weights')
                else:
                    weights = tf.get_variable('proj_weights',
                                              [input_size, output_size])
                # do the appropriate projection
                proj = tf.matmul(inputs, weights)
                # get them into rnn style shape
                proj = tf.unstack(tf.reshape(proj,
                                             [-1, batch_size, output_size]))
                projections.append(proj)
        # now we need to interleave the projections (if we did more than one)
        if len(projections) > 1:
            projections = list(
                itertools.chain.from_iterable(zip(*projections)))
            return projections
        return projections[0]


def sample_net_train(inputs, frame_size_zero=128, cell=None,
                     embedding_size=256, recurrent_tiers=3, upsample_ratio=4,
                     mlp_shape=None):
    """Puts together the SampleNet model for training. This is different from
    putting it together for generating audio, because it's much easier to
    wrangle the RNNs when we know that all the data will be provided in
    advance.

    Args:
        inputs (tensor): the input, raw samples. We expect the shape to be
            `[batch_size, frame_size_zero*num_steps]` where `frame_size_zero`
            is the base (largest) receptive field and `num_steps` is the number
            of steps (of the slowest RNN) we are back-propagating through.
            If this is an integer type, we will convert it to floats and
            posibly embed it. Currently we only support uint8 as an integer
            type and float32 for floating points.
        frame_size_zero (Optional[int]): the size of the input chunks for the
            slowest RNN. We need to be able to divide this by `upsample_ratio`
            `recurrent_tiers` times or we will not have any samples to give
            the faster RNNs. This is also the number of samples the
            autoregressive mlp sees, for convenience (although there is no
            real reason these have to be the same).
        cell (Optional[tf.nn.rnn_cell.RNNCell]): a callable cell which
            constructs the recurrent layers. If unspecified, defaults to GRU
            with 1024 hidden units per layer and 1 layer per tier.
        embedding_size (Optional[int]): size of the embeddings, if the input
            is integers, they are embedded before being passed into the
            autoregressive mlp. If the size is 1, nothing is embedded.
        recurrent_tiers (Optional[int]): how many tiers of recurrent cells.
            Defaults to 3. Each tier operates at a different time-scale.
        upsample_ratio (Optional[int]): how the time-scale changes between
            tiers. At the moment this is a single integer which defines the
            ratio of frequencies between tiers. The clock frequency of tier
            k will the the clock frequency of tier k-1 times upsample_ratio.
            Defaults to 4.
        mlp_shape (Optional[list]): the shape of the autoregressive mlp at the
            final layer. For a description of the format, see
            the `autoregressive_mlp` function.

    Returns:
        tensor: `[batch_size, frame_size_zero*num_steps, mlp_shape[-1]]`
            the output of the network, to be turned into a distribution over
            sample values.
    """
    # defaults
    if cell is None:
        cell = tf.nn.rnn_cell.GRUCell(1024)
    # the first thing to do is sort out the data
    batch_size = inputs.get_shape()[0].value
    if inputs.dtype == tf.uint8:
        float_inputs = tf.cast(inputs, tf.float32) / 128
        float_inputs -= 1.0

        if embedding_size > 1:
            with tf.variable_scope('embeddings'):
                embeddings = tf.get_variable('embedding_matrix',
                                             shape=[256, embedding_size])
                embedded_inputs = tf.nn.embedding_lookup(
                    embeddings, tf.cast(inputs, tf.int32))
        else:
            embedded_inputs = None

    # and now get the RNNs
    current_frame_size = frame_size_zero
    initial_states = []
    final_states = []
    for tier in list(reversed(range(1, recurrent_tiers+1))):
        with tf.variable_scope('tier_{}'.format(tier)):
            print('tier: {}, frame size: {}'.format(tier, current_frame_size))
            rnn_inputs = non_overlapping_frames(float_inputs,
                                                current_frame_size)
            print('         {} frames'.format(len(rnn_inputs)))
            if tier != recurrent_tiers:
                # then we have an output to condition on
                # so we first need to upsample it
                input_size = rnn_outputs[0].get_shape()[1].value
                # first apply the same projection to each group of samples
                rnn_inputs = linear_projection(rnn_inputs,
                                               output_size=input_size,
                                               rate=1,
                                               weightnorm=True,
                                               scope='sample_projection')
                # then upsample the previous outputs
                upsampled = linear_projection(rnn_outputs,
                                              output_size=input_size,
                                              rate=upsample_ratio,
                                              weightnorm=True,
                                              scope='upsample_projection')
                # just a quick sanity check
                assert len(rnn_inputs) == len(upsampled)
                # and stick them together
                rnn_inputs = [prev + samples for prev, samples
                              in zip(upsampled, rnn_inputs)]
            # might be nice to avoid doing this too much
            rnn_inputs = tf.stack(rnn_inputs)
            initial_states.append(cell.zero_state(batch_size, tf.float32))
            rnn_outputs, final_state = tf.nn.dynamic_rnn(
                cell, rnn_inputs, initial_state=initial_states[-1],
                dtype=tf.float32, time_major=True)
            rnn_outputs = tf.unstack(rnn_outputs)
            final_states.append(final_state)
        # reduce the receptive field for the next tier
        current_frame_size //= upsample_ratio
    # upsample the output of the final rnn to give us per-sample conditioning
    # vectors
    rnn_outputs = linear_projection(rnn_outputs, output_size=embedding_size,
                                    rate=current_frame_size*upsample_ratio,
                                    weightnorm=True)
    print('{} rnn outs, {} samples'.format(len(rnn_outputs),
                                           inputs.get_shape()[1].value))

    # and now we can get the autoregressive mlp
    # this is going to make the graph a bit crazy because we are going to have
    # to do this for each sample

    # first step is to get the inputs sorted, we want them batch major,
    # and [batch, time, features]
    if embedded_inputs is None:
        mlp_inputs = tf.expand_dims(float_inputs, -1)
    else:
        mlp_inputs = embedded_inputs
    # mlp_inputs = tf.expand_dims(mlp_inputs, 2)

    # mlp_outputs = []
    # with tf.variable_scope('mlp') as scope:
    #     for i in range(len(inputs)):
    #         # get a window of samples
    #         window = mlp_inputs[:, i:i+frame_size_zero]
    #         # project it
    #         window, = linear_projection([window])
    #         # add to the appropriate conditioning vector
    #         inp = window + rnn_outputs[i]
    #         # and get the mlp
    #         mlp_outputs.append(autoregressive_mlp(inp, shape=mlp_shape))
    #         # if this was the first time, start reusing variables for later
    #         if i == 0:
    #             scope.reuse_variables()
    # return tf.stack(mlp_outputs, name='samplenet_output', axis=1)

    # apply the mlp convolutionally
    # probably the best way to do this is with a loop as per dynamic_rnn
    # with tf.variable_scope('output_tier'):
    #     # first actually pull out the overlapping windows
    #     # this is potentially going to use absurd quantities of memory
    #     mlp_inputs = tf.extract_image_patches(
    #         mlp_inputs, [1, frame_size_zero, 1, 1], [1, 1, 1, 1],
    #         [1, 1, 1, 1],
    #         padding='VALID')
    #     # reshape into an enormous batch
    #     mlp_inputs = tf.reshape(mlp_inputs,
    #                             [-1, mlp_inputs.get_shape()[-1].value])
    #     print('mlp inputs: {}'.format(mlp_inputs.get_shape()))
    #     # and run the mlp
    #     mlp_outputs = autoregressive_mlp(mlp_inputs, shape=mlp_shape)
    #     # now reshape the outputs to what we want: `[batch, time, ...]`
    #     print('mlp_outputs: {}'.format(mlp_outputs.get_shape()))
    #     mlp_outputs = tf.reshape(mlp_outputs,
    #                              [batch_size,
    #                               inputs.get_shape()[1].value-frame_size_zero+1,
    #                               -1])

    with tf.variable_scope('output_tier') as scope:
        #  what appears to be the only feasible solution
        first_mlp_out = autoregressive_mlp(mlp_inputs[:, :frame_size_zero, :],
                                           shape=mlp_shape)
        scope.reuse_variables()

        output_shape = first_mlp_out.get_shape().as_list()
        data_shape = mlp_inputs.get_shape().as_list()
        print(output_shape)

        output_ta = tf.TensorArray(tf.float32,
                                   size=data_shape[1]).write(0, first_mlp_out)

        def apply_mlp(time, outputs, data):
            input_window = data[:, time:time+frame_size_zero, ...]
            input_window.set_shape([data_shape[0],
                                    frame_size_zero,
                                    data_shape[-1]])
            mlp_out = autoregressive_mlp(input_window, shape=mlp_shape)

            # mlp_out.set_shape(output_shape)

            output_ta = outputs.write(time, mlp_out)

            return time+1, output_ta, data

        time_steps = inputs.get_shape()[1]
        time = tf.constant(1, dtype=tf.int32, name='time')
        _, mlp_outputs, __ = tf.while_loop(
            cond=lambda time, *_: time < time_steps-frame_size_zero,
            body=apply_mlp,
            loop_vars=(time, output_ta, mlp_inputs),
            swap_memory=True,
            parallel_iterations=1,)  # TODO: make into param
            # shape_invariants=([1], output_shape, data_shape))

        output_tensor = mlp_outputs.pack()
        print(output_tensor.get_shape())
        output_tensor = tf.reshape(output_tensor,
                                   [data_shape[0],
                                    data_shape[1]-frame_size_zero+1,
                                    output_shape[-1]],
                                   name='sample-net_output')
        return output_tensor


if __name__ == '__main__':
    # load up a model with placeholders and write an events file + graphdef
    # for inspection
    pass
