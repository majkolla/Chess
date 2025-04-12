import numpy 
import random 
import Engine
"""
Thought process: 

Value Network

This part accepts a board state and give us a score as the output. 
The simple implemtation used a basic function that compared the amount of 
pieces and give the value of the position based on that. 

However, in this case we have a neural network that takes in a position and gives an estimation on the 
value of the position. The idea is that the value network is going to work as a classifier giving perhaps something like a 
probility that white wins or a probabilty that black wins (can be represented with a negative probability perhaps)

Policy Network: 
This part accepts a board state as an input and gives a set of probabilities 
representing the probability of a move where higher probability means that the 
probability of the move leading to a dub is higher. 

MCTS: 
    


Notes: 

In the original implementaion we used a 2D list, which is fine, but here 
using a tensor using numpy is going to be much faster and thus we need to rerepresnt the 
board (the game state)

"""

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
        self.state = state         # Instance of the gamestate
        self.parent = parent       # Parent MCTSNode
        self.children = {}         # Dictionary: move -> child node
        self.visit_count = 0
        self.total_value = 0
        self.prior = 0  

    def is_leaf(self) -> bool: 
        return len(self.children) == 0


class MCTS: 
    def __init__(self, neural_net, simulations,):
        self.neural_net = neural_net
        self.simulations = simulations


    def search(self, state): 
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
