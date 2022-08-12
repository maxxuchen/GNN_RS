import torch
import numpy as np

from model.RGGCN import ResidualGatedGCNModel
from es.algorithm.solution import Solution
from es.es_config import net_config
from common.greedy_search import Greedy_Search
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class Match_Policy(Solution):
    def __init__(self):
        self.device = 'cpu'
        self.model = ResidualGatedGCNModel(net_config).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.dim = sum(p.numel() for p in self.model.parameters())

    def act(self, state):
        pass
        # states = [state, state]
        # states = GraphNeuralNetwork.process_state(states, torch.device("cpu"))
        # pdparam = self.model(states)
        # action_pd = Argmax(logits=pdparam)
        # action = action_pd.sample().cpu().squeeze().numpy()
        # return action[0]

    def load(self, path):
        # print("... loading models from %s" % (path))
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        # checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['policy'])

    def save(self, path):
        torch.save({
            'policy': self.model.state_dict()
        }, path)

    def get_params(self):
        params = parameters_to_vector(self.model.parameters()).numpy()
        assert params.dtype == np.float32
        return params

    def set_params(self, params):
        assert params.dtype == np.float32
        vector_to_parameters(torch.tensor(params), self.model.parameters())

    def size(self):
        return self.dim

    def evaluate(self, instance):
        """
        Given a instance, use the current model to solve the instance and get its score
        :param instance: a path to one instance, graph = instance['graph'], MWM = instance['MWM']
        :return: score of evaluation
        """
        _, weights = Greedy_Search(self.model, instance)

        score = - weights  # we want minimize (- weights)
        episodes = 1
        transitions = 1
        return score, episodes, transitions
