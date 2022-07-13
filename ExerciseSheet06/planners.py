import torch
import numpy as np

class Planner:
    def predict(self, model, actions, observation):
        observations = []
        for t in range(self._horizon):
            if len(actions[t].shape) > 1:
                inp = torch.cat([observation, actions[t]], dim=1)
            else:
                inp = torch.cat([observation, actions[t]])
            observation = model.forward(inp)
            observations.append(observation)
        return observations

    def __call__(self, model, observation):
        pass


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
        horizon=50,
        num_inference_cycles=2,
        num_predictions=50,
        num_elites=5,
        num_keep_elites=2,
        criterion=torch.nn.MSELoss(),
        policy_handler=lambda x: x,
        var=0.2,
        alpha = 0.1
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
        # self._dist = torch.distributions.MultivariateNormal(
        #     torch.zeros(self._action_size), torch.eye(self._action_size)
        # )
        self._dist = torch.distributions.MultivariateNormal(
            self._mu, self._covariance_init
        )
        self._last_actions = None
        self.alpha = alpha
        

    def __call__(self, model, observation):
        old_elite_actions = torch.tensor([])
        # self._dist = torch.distributions.MultivariateNormal(
        #             self._mu, self._var
        #         )
        for _ in range(self._num_inference_cycles):
            with torch.no_grad():
                # TODO: implement CEM
                # actions = self._policy_handler(self._dist.sample(torch.Size([self._num_predictions])))
                actions = self._policy_handler(self._dist.sample(torch.Size([self._num_predictions])))
                observations = torch.stack([torch.stack(self.predict(model, action, observation)) for action in actions])
                # observations = torch.stack(observations)#[None]
                # loss = torch.tensor([self._criterion(obs) for obs in observations])
                loss = self._criterion(observations) 
            
                #elite_idxs = np.array(loss).argsort()[: self.num_elites]
                elite_idxs = loss.argsort()[: self._num_elites]
                elite_actions = actions[elite_idxs]
                # take _num_keep_elites from previous run and concat with new elites
                # old_elites_selection = old_elite_actions[torch.randperm(len(old_elite_actions))[:self._num_keep_elites]]
                old_elites_selection = old_elite_actions[:self._num_keep_elites]
                elite_actions = torch.cat((elite_actions,old_elites_selection),0)
                old_elite_actions = elite_actions
                new_mean = elite_actions.mean(axis=0)
                new_std = elite_actions.std(axis=0)
                # Momentum term
                self._mu= (1 - self.alpha) * new_mean + self.alpha * self._mu
                self._var  = (1 - self.alpha) * new_std + self.alpha * self._var
                # old_mean = self._mu
                # old_var  = self._var
                # set dist
                self._dist = torch.distributions.MultivariateNormal(
                    self._mu, torch.stack([torch.diag(actions) for actions in self._var])
                )
            # self._update_bounds(like_levine=self.like_levine)
            

        # Policy has been optimized; this optimized policy is now propagated
        # once more in forward direction in order to generate the final
        # observations to be returned
        actions = actions[0][None]
        actions = actions.permute(1, 0, 2)  # [time, batch, action]
        with torch.no_grad():
            observations = self.predict(model, actions[:, 0, :], observation)

        with torch.no_grad():
            # Shift means for one time step
            self._mu[:-1] = self._mu[1:].clone()
            # Reset the variance
            self._var = self._var_init.clone() #unclear
            # Shift elites to keep for one time step
            # self._last_actions[:, :-1] = self._last_actions[:, 1:].clone() #unclear

        return actions[:, 0, :], observations
