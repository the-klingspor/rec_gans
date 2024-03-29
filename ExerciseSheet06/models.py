import torch


class SimpleModel:

    def __init__(self, env):
        self.agent_pos_upper_limits = torch.tensor(env.agent_pos_upper_limits).float()
        self.agent_pos_lower_limits = torch.tensor(env.agent_pos_lower_limits).float()
        # Scaling of action effect
        self.action_factor = env.action_factor

    def clip_positions(self, agent_pos):
        agent_pos = torch.where(agent_pos < self.agent_pos_lower_limits, self.agent_pos_lower_limits, agent_pos)
        agent_pos = torch.where(agent_pos > self.agent_pos_upper_limits, self.agent_pos_upper_limits, agent_pos)
        return agent_pos

    def forward(self, x):
        x = x.float()
        with torch.no_grad():
            agent_pos = x[..., :2]
            actions = x[..., 2:]
            agent_pos = agent_pos + self.action_factor * actions
            agent_pos = self.clip_positions(agent_pos)
        return agent_pos


class NeuralNetworkModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetworkModel, self).__init__()
        # TODO: implement me
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=True),
            #torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            #torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=True),
            torch.nn.Tanh()
        )

    def forward(self, x):
        # TODO: implement me
        Observation_High = torch.tensor([1.5, 1.5, 5., 5., 3.14, 5.,1., 1.])  # 6 observations + 2 actions 
        Observation_Low = torch.tensor([-1.5, -1.5, -5., -5., -3.14, -5.,-1., -1.])

        # normalize variance, mean is already zero
        normal_input = x / (Observation_High - Observation_Low)
        model_out = self.model(normal_input)

        # return predicted state, model learns residual change of environment 
        # actions are at the end actually
        return model_out + x[..., :-2]
