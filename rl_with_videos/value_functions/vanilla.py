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
                                  **kwargs):
    input_shapes = (observation_shape, action_shape)
    preprocessors = (observation_preprocessor, None)
    print("preprocessors:", preprocessors)

    q =  feedforward_model(
        input_shapes,
        *args,
        output_size=1,
        preprocessors=preprocessors,
        name=name,
        **kwargs)

    domain_shift_model = None
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
    return q, domain_shift_model

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
