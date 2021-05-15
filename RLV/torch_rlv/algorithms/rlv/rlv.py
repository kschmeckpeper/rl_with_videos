import torch.nn as nn
from RLV.torch_rlv.models.base_model import InverseModelNetwork
from RLV.torch_rlv.buffer.provided_replay_pools.adapter_acrobot import get_acrobot_observations_actions

# get Acrobot Data
inputs, target = get_acrobot_observations_actions()

# Create NN
inverse_model_acrobot = InverseModelNetwork(beta=0.0003, input_dims=inputs.shape)

# Compute Output
output = inverse_model_acrobot(inputs)
criterion = nn.MSELoss()
loss = criterion(output, target)

iterations = 5000

# Backprop - training loop
for _ in range(0, iterations):
    inverse_model_acrobot.optimizer.zero_grad()   # zero the gradient buffers
    output = inverse_model_acrobot(inputs)
    loss = criterion(output, target)
    loss.backward()
    inverse_model_acrobot.optimizer.step()    # Does the update
    if _ % 200 == 0:
        print(f"Iteration: {_} , Loss: {loss}")

print(f"Iteration: {iterations} , Loss: {loss}")

