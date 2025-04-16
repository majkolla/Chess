import numpy 
import random 
import Engine 
import copy 
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F


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
Basic idea is to start with an empty tree, then use MCTS to build up a portion of the game tree by running 
a number of simulations, where each simulation adds a node to the tree.  


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

    def predict(self, state : Engine.GameState): 
        """
        We represent the board as a 2D tensor (matrix): 
        we return policy, value as a tuplem 
        """
        moves = state.get_all_valid_moves()
        if moves: 
            prob = 1.0 / len(moves)
            policy = {move: prob for move in moves}
        else: 
            policy = {}
        value = random.uniform(-1,1)
        return policy, value 

    def train(self, training_data): 
        """ 
        Here we want to train the neural network on the training data
        Im thinking: training_data = (state :  state_tensor, move_probabilities, gam_outcome)

        """
        # we implemet the training loop maybe by backpropagation or something. 
        
        pass 

    def save_model(self, file_path):
        # Save the model's state dictionary
        torch.save(self.model.state_dict(), file_path)
        print("Model saved to", file_path)

    def load_model(self, file_path):
        # Load the model's state dictionary
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()  # Set the model to evaluation mode
        print("Model loaded from", file_path)

class MCTSNode: 
    """
    Monte carlo tree search data structure

    The point here is that each node representrs a game state
    
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
    """
    Monte carlo tree search implementation
    """
    def __init__(self, neural_net, simulations,):
        self.neural_net = neural_net
        self.simulations = simulations

    def get_next_state(self, state : Engine.GameState , move):
        new_state = copy.deepcopy(state)
        new_state.make_move(move)
        return new_state

    def search(self, state): 
        """
        Perform MCTS starting from the state
        we return a map of moves to prob 
        """
        root = MCTSNode(state)

        for _ in range(self.simulations):
            self.simulate(root)

        # After all simulations, compute move probabilities from visit counts.
        move_visits = {move: child.visit_count for move, child in root.children.items()}
        total_visits = sum(move_visits.values())
        if total_visits > 0:
            move_probs = {move: count / total_visits for move, count in move_visits.items()}
        else:
            move_probs = {}
        return move_probs
    
    def simulate(self, node): 
        """
        Here we run one sim starting from a node 
        """
        outcome = self.get_outcome(node.state)
        if outcome is not None:
            # Terminal state reached: return outcome
            return outcome

        if node.is_leaf():
            # Expand leaf: use neural network to obtain move probabilities and value
            policy, value = self.neural_net.predict(node.state)
            # Expand children nodes based on the policy
            for move, p in policy.items():
                next_state = self.get_next_state(node.state, move)
                child_node = MCTSNode(next_state, parent=node)
                child_node.prior = p
                node.children[move] = child_node
            return value
        else:
            # Select the child with the highest UCB score and simulate down from it
            best_child = self.select_child(node)
            value = self.simulate(best_child)
            self.backpropagate(best_child, value)
            return value
        

    def select_child(self, node): 
        """
        we select child based of the UCB formula: 
        score = Q / (N) + c_puct * prior * sqrt(parent_visits) / (1 + N)
        Q - total value of the node 
        N - visit count of the node
        """

        c_puct = 1.0
        best_score = -float("inf")
        best_child = None
        
        for move, child in node.children.items():
            if child.visit_count == 0:
                ucb = c_puct * child.prior * (node.visit_count ** 0.5)
            else:
                ucb = (child.total_value / child.visit_count +
                       c_puct * child.prior * (node.visit_count ** 0.5) / (1 + child.visit_count))
            if ucb > best_score:
                best_score = ucb
                best_child = child
        return best_child
    

    def backpropagate(self, node, value): 
        """
        Here we propagate the sim value up the tree

        the alt the sign for each move so it becomes correct with the turns and stuff!
        """
        while node is not None: 
            node.visit_count += 1
            node.total_value += value
            value = -value  # Switch perspective for the opponent
            node = node.parent


    def get_outcome(self, state : Engine.GameState): 
        """ 
        Encoding: 
        1 if white won 
        -1 if black won 
        0 if draw 
        """
        if state.check_mate: 
            if state.white_to_move: 
                return -1
            else: 
                return 1 
        elif state.stale_mate: 
            return 0 
        return None #Game not over! python is so nice we aint need no encoding for that lol
    

def self_play(neural_net : NeuraNet, simulations): 
    """
    play a game by using the self play with MCTS moves. 
    Each MCTS output (state, move prob) is saved for the training

    alg: 
    - run mcts to get move probs 
    - select a move 
    - save the state and move prob for trainging
    
    """
    training : list = []
    state = Engine.GameState()
    mcts = MCTS(neural_net, simulations)

    # While game not over (assuming check_mate and stale_mate are flags in the state)
    while not state.check_mate and not state.stale_mate:
        # Run MCTS to determine move probabilities for current state
        move_probs = mcts.search(state)
        if not move_probs:
            # No moves available; break out of the loop
            break
        
        # Choose a move – here we use a weighted choice based on move probabilities.
        moves = list(move_probs.keys())
        probs = list(move_probs.values())
        chosen_move = random.choices(moves, weights=probs, k=1)[0]
        
        # Save current state representation and move probs.
        training.append((state.get_current_state(), move_probs, None))
        
        # Make the move
        state.make_move(chosen_move)

    # Now the game is over and we assign the game outcome to each training example 
    outcome = mcts.get_outcome(state)
    training = [(s, p, outcome) for (s, p, _) in training]
    return training


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Example architecture: a simple conv + fc network.
        # Input channels: 13 (e.g., 12 channels for pieces + 1 for side-to-move)
        self.conv = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=3, padding=1)
        self.fc_policy = nn.Linear(64 * 8 * 8, 4672) 
        self.fc_value = nn.Linear(64 * 8 * 8, 1)
    
    def forward(self, x):
        # x is expected to have shape [batch_size, 13, 8, 8]
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        policy_logits = self.fc_policy(x)
        value = torch.tanh(self.fc_value(x))
        return policy_logits, value


if __name__ == "__main__": 
    nn = NeuraNet()

    sims = 50
    training_data = self_play(nn, sims) 
    
    print("Generated {} training samples.".format(len(training_data)))
    nn.train(training_data)

    nn.save_model("chess_model.pth")
