from rl_with_videos.models.feedforward import feedforward_model
from rl_with_videos.utils.gradient_reversal import gradient_reversal
from rl_with_videos.utils.keras import PicklableKerasModel
import tensorflow as tf

def create_feedforward_Q_function(observation_shape,
                                  action_shape,
                                  *args,
                                  observation_preprocessor=None,
                                  domain_shift=False,
                                  name='feedforward_Q',
                                  auxiliary_loss=False,
                                  **kwargs):
    input_shapes = (observation_shape, action_shape)
    preprocessors = (observation_preprocessor, None)
    print("preprocessors:", preprocessors)
    advantage_model = None
    value_model = None

    q =  feedforward_model(
        input_shapes,
        *args,
        output_size=1,
        preprocessors=preprocessors,
        name=name,
        **kwargs)

    domain_shift_model = None
    auxiliary_dynamics_model = None
    if domain_shift:

        domain_shift_model =  feedforward_model(
            (input_shapes[0],),
            *args,
            output_size=1,
#            reverse_gradients=True,
            preprocessors=(preprocessors[0],),
            name=name + "_domain_shift",
            output_activation='sigmoid',
            **kwargs)
        if auxiliary_loss:
            output_activation='linear'
            activation='relu'
            print("args", args)
            print("kwargs", kwargs)
            hidden_layer_sizes = kwargs['hidden_layer_sizes']
            output_size = 6*6*10


            inputs = [
                tf.keras.layers.Input(shape=input_shape)
                for input_shape in input_shapes
            ]
            print("\n\n\nauxiliary loss\n\n\n")
            print("inputs shape:", input_shapes[0], input_shapes[1])
            preprocessed_inputs = [
                preprocessor(input_) if preprocessor is not None else input_
                for preprocessor, input_ in zip(preprocessors, inputs)
            ]

            concatenated = tf.keras.layers.Lambda(
                lambda x: tf.concat(x, axis=-1)
            )(preprocessed_inputs)

            out = concatenated
            for units in hidden_layer_sizes:
                out = tf.keras.layers.Dense(
                    units, *args, activation=activation,
                )(out)

            out = tf.keras.layers.Dense(
                output_size, *args, activation=output_activation,
            )(out)
            out = tf.keras.layers.Lambda(lambda out: tf.Print(out, [tf.shape(out)], "out shape before reshape"))(out)
            out = tf.keras.layers.Reshape((6, 6, 10))(out)
            print("out shape", out.shape)
            out = tf.keras.layers.Conv2D(32, 3, padding='same', activation=activation)(out)

            print("out shape", out.shape)
            out = tf.keras.layers.Conv2DTranspose(32, 3, strides=(2, 2), padding="same", activation=activation)(out)
            out = tf.keras.layers.Conv2DTranspose(32, 3, strides=(2, 2), padding="same", activation=activation)(out)
            out = tf.keras.layers.Conv2DTranspose(32, 3, strides=(2, 2), padding="same", activation=activation)(out)
            print("out shape", out.shape)
            reshaped_im_0 = tf.keras.layers.Reshape((48, 48, 3))(inputs[0])
            out = tf.keras.layers.Lambda(
                lambda x: tf.concat(x, axis=-1)
            )([out, reshaped_im_0])
            

            out = tf.keras.layers.Conv2D(32, 3, padding='same', activation=activation)(out)
            out = tf.keras.layers.Conv2D(3, 3, padding='same', activation="sigmoid")(out)
            print("out shape", out.shape)
            out = tf.keras.layers.Lambda(lambda out: tf.Print(out, [tf.shape(out)], "out shape before reshape2"))(out)
            out = tf.keras.layers.Reshape((48*48*3,))(out)
            print("out shape", out.shape)
            auxiliary_dynamics_model = PicklableKerasModel(inputs, out, name=name+"_auxiliary_model")
    return q, domain_shift_model, auxiliary_dynamics_model, advantage_model, value_model

def create_feedforward_V_function(observation_shape,
                                  *args,
                                  observation_preprocessor=None,
                                  name='feedforward_V',
                                  **kwargs):
    input_shapes = (observation_shape, )
    preprocessors = (observation_preprocessor, None)
    return feedforward_model(
        input_shapes,
        *args,
        output_size=1,
        preprocessors=preprocessors,
        **kwargs)
