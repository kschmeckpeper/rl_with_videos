import tensorflow as tf


from rl_with_videos.utils.keras import PicklableKerasModel
from rl_with_videos.utils.gradient_reversal import gradient_reversal

def feedforward_model(input_shapes,
                      output_size,
                      hidden_layer_sizes,
                      activation='relu',
                      reverse_gradients=False,
                      stop_gradients=False,
                      output_activation='linear',
                      preprocessors=None,
                      name='feedforward_model',
                      *args,
                      **kwargs):
    inputs = [
        tf.keras.layers.Input(shape=input_shape)
        for input_shape in input_shapes
    ]
    print("name:", name)
    print("inputs:", inputs)
    if preprocessors is None:
        preprocessors = (None, ) * len(inputs)

    preprocessed_inputs = [
        preprocessor(input_) if preprocessor is not None else input_
        for preprocessor, input_ in zip(preprocessors, inputs)
    ]

    concatenated = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )(preprocessed_inputs)

    if reverse_gradients:
        print("concatenated:", concatenated)
        concatenated = tf.keras.layers.Lambda(lambda t: gradient_reversal(t))(concatenated)
        print("after:", concatenated)
    if stop_gradients:
        print("stopping gradient to preprocessor")
        concatenated = tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x))(concatenated)

    out = concatenated
    for units in hidden_layer_sizes:
        out = tf.keras.layers.Dense(
            units, *args, activation=activation, **kwargs
        )(out)

    out = tf.keras.layers.Dense(
        output_size, *args, activation=output_activation, **kwargs
    )(out)

    model = PicklableKerasModel(inputs, out, name=name)

    return model
