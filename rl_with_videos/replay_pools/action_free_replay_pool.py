from gym.spaces import Dict

from .flexible_replay_pool import FlexibleReplayPool
from .simple_replay_pool import normalize_observation_fields

class ActionFreeReplayPool(FlexibleReplayPool):
    def __init__(self,
                 observation_space,
                 action_space,
                 data_path=None,
                 *args,
                 extra_fields=None,
                 remove_rewards=False,
                 use_ground_truth_actions=False,
                 max_demo_length=-1,
                 **kwargs):
        extra_fields = extra_fields or {}
#        action_space = environment.action_space
#        assert isinstance(observation_space, Dict), observation_space

#        self._environment = environment
        self._observation_space = observation_space
        self._action_space = action_space
        print("self._observation_space", self._observation_space)

        observation_fields = normalize_observation_fields(observation_space)
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have
        # to worry about termination conditions.
        observation_fields.update({
            'next_' + key: value
            for key, value in observation_fields.items()
        })

        fields = {
            **observation_fields,
            **{
                'actions': {
                    'shape': self._action_space.shape,
                    'dtype': 'float32'
                },
                'rewards': {
                    'shape': (1, ),
                    'dtype': 'float32'
                },
                # self.terminals[i] = a terminal was received at time i
                'terminals': {
                    'shape': (1, ),
                    'dtype': 'bool'
                },
            }
        }

        super(ActionFreeReplayPool, self).__init__(
            *args, fields_attrs=fields, **kwargs)
        print("about to load replay pool")
        self.load_experience(data_path)
        print("loaded experience of size:", self.size) 
        if not use_ground_truth_actions:
            self.fields.pop('actions')
#        self.fields_flat.pop(('actions',))

        if False:
            num_images = 10000
            obs = self.fields["observations"]
            print("obs:", obs.shape, type(obs))
            import cv2
            import numpy as np
            base_out_path = "/scratch/karls/pac_cyclegan_10k"
            for i in range(num_images):
                im = np.reshape(obs[i*num_images//obs.shape[0], :], (48, 48, 3)) * 255
                if i % 10 == 0:
                    path = base_out_path + "/testB/"
                else:
                    path = base_out_path + "/trainB/"
                path = path + "{:06d}.jpg".format(i)

                cv2.imwrite(path, im)

            exit()

        if max_demo_length != -1 and max_demo_length < self.fields['observations'].shape[0] and max_demo_length < self._size:
            print("going from size {} or {} to size {}".format(self.fields['observations'].shape[0], self._size, max_demo_length))
            for k in self.fields.keys():
                self.fields[k] = self.fields[k][self._size-max_demo_length:self._size]
            self._size = max_demo_length

        if remove_rewards:
            self.fields.pop('rewards')
#            self.fields_flat.pop(('rewards',))


    """ The action-free replay pool should not be added to during runtime
    This removes the methods that were inherited from FlexibleReplayPool
    """
    def add_sample(self, sample):
        raise NotImplementedError

    def add_path(self, path):
        raise NotImplementedError
