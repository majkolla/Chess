import numpy 
import random 
import Engine


class NeuraNet:
    """
    
    """
    def __init__(self): 
        pass 

    def predict(self, state): 
        """
        We represent the board as a 2D tensor (matrix): 
        we return policy, value as a tuplem 
        """
        pass 

    def train(self, training_data): 
        """ 
        Here we want to train the neural network on the training data
        Im thinking: training_data = (state : 2D tensor, move_probabilities, gam_outcome)

        """
        pass 



class MCTSNode: 
    """
    Monte carlo tree search data structure
    """
    def __init__(self, state, parent=None):
        pass 

    def is_leaf(self) -> bool: 
        return len(self.children) == 0


class MCTS: 
    def __init__(self):
        pass

    def search(): 
        pass 

    def simulate(self, node): 
        pass 
    def select_child(self, node): 
        pass 
    def backpropagate(self, node, value): 
        pass 
    def get_outcome(self, state): 
        pass 

def self_play(neural_net : NeuraNet, simulations): 
    """
    collect data by doing MCTS 
    """
