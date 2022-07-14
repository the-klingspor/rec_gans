import torch
import numpy as np
import gym



class Planner:
    def predict(self, model, actions, observation):
        if len(actions.shape) == 3:
            # more than one prediction ->
            # change shape from [num_predictions, horizon, n_actions]
            # to [horizon, num_predictions, n_actions]
            actions = actions.transpose(0, 1)
            observation = observation.repeat(actions.shape[1], 1)
        observations = []
        for t in range(self._horizon):
            if len(actions[t].shape) > 1:
                inp = torch.cat([observation, actions[t]], dim=1)
            else:
                inp = torch.cat([observation, actions[t]])
            observation = model.forward(inp)
            observations.append(observation)
        observations = torch.stack(observations)
        if len(actions.shape) == 3:
            observations = observations.transpose(0, 1)  # back to original shape
        return observations

    def __call__(self, model, observation):
        pass
    
    def predict_env(self, actions,old_actions,seed):
        # initalize new env with same seed and execute all actions that happend so far
        # to predict new observations
        observations = []
        env_new = gym.make("LunarLander-v2",continuous=True,  enable_wind=False)
        for num_pred in range(actions.shape[0]):
            observation_old = env_new.reset(seed=seed)
            for action in old_actions:
                observation__, _, done, _ = env_new.step(action.numpy())

            observation = []
            for t in range(self._horizon):
                observation_single, _, done, _ = env_new.step( actions[num_pred][t].numpy())
                observation.append(observation_single)
            observations.append(observation)
        observations = torch.tensor(observations)
        return observations[:,:,:-2]


class RandomPlanner(Planner):
    def __init__(
        self,
        action_size=2,
        horizon=20,
    ):
        Planner.__init__(self)
        self._horizon = horizon
        self._action_size = action_size

    def __call__(self, model, observation):
        actions = torch.rand([self._horizon, self._action_size]) * 2 - 1
        with torch.no_grad():
            observations = self.predict(model, actions, observation)
        return actions, observations


class CrossEntropyMethod(Planner):
    def __init__(
        self,
        action_size=2,
        horizon=5,
        num_inference_cycles=20,
        num_predictions=50,
        num_elites=5,
        num_keep_elites=2,
        criterion=torch.nn.MSELoss(),
        policy_handler=lambda x: x,
        var=0.2,
        alpha=0.01
    ):
        Planner.__init__(self)
        self._action_size = action_size
        self._horizon = horizon
        self._num_inference_cycles = num_inference_cycles
        self._num_predictions = num_predictions
        self._num_elites = num_elites
        self._num_keep_elites = num_keep_elites
        self._criterion = criterion
        self._policy_handler = policy_handler

        self._mu = torch.zeros([self._horizon, self._action_size])
        self._var_init = var * torch.ones([self._horizon, self._action_size])
        self._covariance_init = var * torch.eye(self._action_size,self._action_size).unsqueeze(1).permute(0,2,1).repeat(1,1,self._horizon).permute(2,0,1)
        self._var = self._var_init.clone().detach()
        self._dist = torch.distributions.MultivariateNormal(
            self._mu, self._covariance_init
        )
        self._last_actions = None
        self.alpha = alpha
        
    def __call__(self, model, observation,old_actions,env_seed):
        old_elite_actions = torch.tensor([])
        self._dist = torch.distributions.MultivariateNormal(
                    self._mu, torch.stack([torch.diag(actions) for actions in self._var])
                )
        # print("new cycle")
        for _ in range(self._num_inference_cycles):
            with torch.no_grad():
                # TODO: implement CEM
                actions = self._policy_handler(self._dist.sample(torch.Size([self._num_predictions])))

                # Neural Net Observations
                # observations = self.predict(model, actions, observation)
                # Environment generated Observations
                observations = self.predict_env(actions,old_actions,env_seed)

                # Loss Batch calculation (with original loss from exercise sheet)
                # loss = self._criterion(observations) #batch
                # Loss Loop calculation
                loss = torch.stack([self._criterion(x[None]) for x in observations])
            

                elite_idxs = loss.argsort()[: self._num_elites]
                elite_actions = actions[elite_idxs]
                # keep best elites
                old_elites_selection = old_elite_actions[:self._num_keep_elites]
                elite_actions = torch.cat((elite_actions, old_elites_selection), 0)
                old_elite_actions = elite_actions
                new_mean = elite_actions.mean(axis=0)
                new_var = elite_actions.std(axis=0)**2
                # Momentum term - alpha very small, but still needed for stability (matrix becomes non psd)
                self._mu = (1 - self.alpha) * new_mean + self.alpha * self._mu
                self._var = (1 - self.alpha) * new_var + self.alpha * self._var

                # set dist
                self._dist = torch.distributions.MultivariateNormal(
                    self._mu, torch.stack([torch.diag(actions) for actions in self._var])
                )
            
            

        # Policy has been optimized; this optimized policy is now propagated
        # once more in forward direction in order to generate the final
        # observations to be returned
        actions = actions[0][None]
        with torch.no_grad():
            # NN Observations
            # observations = self.predict(model, actions[0, :, :], observation)
            # Env Observations
            obs = self.predict_env(actions,old_actions,env_seed)

        with torch.no_grad():
            # Shift means for one time step
            self._mu[:-1] = self._mu[1:].clone()
            # Reset the variance
            self._var = self._var_init.clone() #unclear
            # Shift elites to keep for one time step
            # self._last_actions[:, :-1] = self._last_actions[:, 1:].clone() #unclear

        actions = actions.permute(1, 0, 2)  # [time, batch, action]
        return actions[:, 0, :], observations
