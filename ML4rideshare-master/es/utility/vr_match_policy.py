import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from models.Bipartite_RGGCN import Bipartite_RGGCNModel
from bc.bc_config import NUM_VEHICLE
from es.algorithm.solution import Solution
from es.es_config import VR_net_config as net_config
from es.utility.matcher import RR_Matcher, VR_Matcher


class VR_Match_Policy(Solution):
    def __init__(self):
        self.model = Bipartite_RGGCNModel(net_config).to('cpu')
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
        self.model.load_state_dict(checkpoint)
        # checkpoint = torch.load(path)
        # self.model.load_state_dict(checkpoint['policy'])

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        # torch.save({
        #     'policy': self.model.state_dict()
        # }, path)

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
        Evaluate vr_policy with fixed rr_policy
        rr_policy: nx.algorithms.matching.max_weight_matching, heu
        vr_policy: self.model, GNN model
        :return: score of evaluation
        """
        rr_matcher = RR_Matcher('heu')
        vr_matcher = VR_Matcher('nn', self.model)

        # Create simulator for eval
        from env.simulator import Simulator
        env = Simulator(NUM_VEHICLE, instance)
        # we may need seed for the simulator to get same day
        env.reset()
        done = False

        Assigned_Request, Loss_Request, Distance_Driven, Waiting_Time, Revenue = 0, 0, 0, 0, 0
        while not done:
            dispatch_action = {}
            rr_graph = env.get_rr_match_graph()
            if rr_graph is not None:
                _, rr_decision = rr_matcher.get_rr_match_decision(rr_graph)
                env.do_rr_match(rr_decision)

                vr_graph = env.get_vr_match_graph()
                if vr_graph is not None and len(vr_graph.edges) > 0:
                    _, vr_decision = vr_matcher.get_vr_match_decision(vr_graph)
                    dispatch_action = env.do_vr_match(vr_decision)

            done, assigned_request, loss_request, distance_driven, waiting_time, revenue = env.step(dispatch_action)
            # do not count the first ten step to avoid cold start
            if env.timestep > 9:
                Assigned_Request += assigned_request
                Loss_Request += loss_request
                Distance_Driven += distance_driven
                Waiting_Time += waiting_time
                Revenue += revenue

        score = - Revenue  # we want minimize (- weights)
        episodes = 1
        transitions = 1
        return score, episodes, transitions
