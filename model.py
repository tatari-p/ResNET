import tensorflow as tf
from tensorflow import layers


def create_resnet_block(x, filters, training, activation=tf.nn.elu, skip_connection=False):
    h = layers.conv2d(x, filters, kernel_size=3, activation=None, padding="SAME", bias_initializer=None)
    h = layers.batch_normalization(h, momentum=0.9, scale=True, fused=True, training=training)
    h = activation(h)
    h = layers.conv2d(h, filters, kernel_size=3, activation=None, padding="SAME", bias_initializer=None)

    if not skip_connection:
        h = h+x

    h = layers.batch_normalization(h, momentum=0.9, scale=True, fused=True, training=training)
    h = activation(h)

    return h


def create_bottleneck_block(x, filters, training, activation=tf.nn.elu, skip_connection=False):
    h = layers.conv2d(x, filters/4, kernel_size=1, activation=None, padding="SAME", bias_initializer=None)
    h = layers.batch_normalization(h, momentum=0.9, scale=True, fused=True, training=training)
    h = activation(h)
    h = layers.conv2d(h, filters/4, kernel_size=3, activation=None, padding="SAME", bias_initializer=None)
    h = layers.batch_normalization(h, momentum=0.9, scale=True, fused=True, training=training)
    h = activation(h)
    h = layers.conv2d(h, filters, kernel_size=1, activation=None, padding="SAME", bias_initializer=None)

    if not skip_connection:
        h = h+x

    h = tf.layers.batch_normalization(h, momentum=0.9, scale=True, fused=True, training=training)
    h = activation(h)

    return h


def downsample(x, filters, training, activation=tf.nn.elu):
    h = layers.conv2d(x, filters, kernel_size=3, strides=2, activation=None, padding="SAME")
    h = layers.batch_normalization(h, momentum=0.9, scale=True, fused=True, training=training)
    h = activation(h)

    return h


def create_resnet(x, num_blocks, num_outputs, training, activation=tf.nn.elu, output_activation=None, bottleneck=False, name=None, reuse=False):
    create_block = create_bottleneck_block if bottleneck else create_resnet_block
    unit = 256 if bottleneck else 64
    name = "ResNET" if name is None else name

    with tf.variable_scope(name, reuse=reuse):
        b = layers.conv2d(x, 64, kernel_size=7, strides=2, activation=activation, padding="SAME")
        b = layers.max_pooling2d(b, pool_size=3, strides=2, padding="SAME")

        for i, num_repeats in enumerate(num_blocks):
            b = create_block(b, unit*2**i, training, activation, skip_connection=True)

            for _ in range(num_repeats-1):
                b = create_block(b, unit*2**i, training, activation)

            b = downsample(b, unit*2**(i+1), training)

        # use global average pooling
        b = layers.conv2d(b, num_outputs, kernel_size=1, activation=None, padding="SAME")
        fts = tf.reduce_mean(b, [1, 2])

        if output_activation is not None:
            fts = output_activation(fts)

    return fts


def create_resnet_18(x, num_outputs, training, activation=tf.nn.elu, output_activation=None, name=None, reuse=False):
    name = "ResNET-18" if name is None else name

    return create_resnet(x, [2, 2, 2, 2], num_outputs, training, activation, output_activation, False, name, reuse)


def create_resnet_34(x, num_outputs, training, activation=tf.nn.elu, output_activation=None, name=None, reuse=False):
    name = "ResNET-34" if name is None else name

    return create_resnet(x, [3, 4, 6, 3], num_outputs, training, activation, output_activation, False, name, reuse)


def create_resnet_50(x, num_outputs, training, activation=tf.nn.elu, output_activation=None, name=None, reuse=False):
    name = "ResNET-50" if name is None else name

    return create_resnet(x, [3, 4, 6, 3], num_outputs, training, activation, output_activation, True, name, reuse)


def create_resnet_101(x, num_outputs, training, activation=tf.nn.elu, output_activation=None, name=None, reuse=False):
    name = "ResNET-101" if name is None else name

    return create_resnet(x, [3, 4, 23, 3], num_outputs, training, activation, output_activation, True, name, reuse)


def create_resnet_152(x, num_outputs, training, activation=tf.nn.elu, output_activation=None, name=None, reuse=False):
    name = "ResNET-152" if name is None else name

    return create_resnet(x, [3, 8, 36, 3], num_outputs, training, activation, output_activation, True, name, reuse)

