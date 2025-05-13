import numpy as np
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


def move_to_index(mv: Engine.move) -> int:
    return ((mv.start_row * 8 + mv.start_col) * 64) + (mv.end_row * 8 + mv.end_col)

def index_to_move(idx: int, state: Engine.GameState) -> Engine.move | None:
    """Return the move object matching index or None if it is illegal in the given state"""
    sr, sc = divmod(idx // 64, 8)
    er, ec = divmod(idx % 64, 8)
    template = Engine.move((sr, sc), (er, ec), state.board)
    for mv in state.get_all_valid_moves():
        if mv == template:
            return mv
    return None



# ────────────────────────────────────────────────────────────────────────────────
class NeuraNet:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = ChessNet().to(self.device)
        self.opt    = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()

    @torch.no_grad()
    def predict(self, state: Engine.GameState) -> tuple[dict[Engine.move, float], float]:
        board = torch.tensor(state.get_current_state(), dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(board)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        legal_moves = state.get_all_valid_moves()
        if not legal_moves:
            return {}, 0.0

        move_probs: dict[Engine.move, float] = {}
        total = 0.0
        for mv in legal_moves:
            p = probs[move_to_index(mv)]
            move_probs[mv] = p
            total += p
        if total == 0:
            move_probs = {m: 1/len(legal_moves) for m in legal_moves}
        else:
            move_probs = {m: p/total for m, p in move_probs.items()}
        return move_probs, float(value.item())

    def train(self, samples: list[tuple[np.ndarray, dict[Engine.move, float], int]],
              epochs: int = 5, batch_size: int = 32):
        NUM_MOVES = 8**4
        if not samples:
            return
        boards , policies , values = [], [], []
        for b, pi, v in samples:
            boards.append(torch.tensor(b, dtype=torch.float32))
            vec = np.zeros(NUM_MOVES, dtype=np.float32)
            for mv, p in pi.items():
                vec[move_to_index(mv)] = p
            policies.append(torch.tensor(vec, dtype=torch.float32))
            values.append(float(v))
        boards   = torch.stack(boards).to(self.device)
        policies = torch.stack(policies).to(self.device)
        values   = torch.tensor(values, dtype=torch.float32).to(self.device)

        for epoch in range(epochs):
            perm = torch.randperm(len(samples), device=self.device)
            boards, policies, values = boards[perm], policies[perm], values[perm]
            for start in range(0, len(samples), batch_size):
                end = start + batch_size
                xb, pb, vb = boards[start:end], policies[start:end], values[start:end]
                self.opt.zero_grad()
                logits, v_pred = self.model(xb)
                loss_p = self.kl(F.log_softmax(logits, dim=1), pb)
                loss_v = self.mse(v_pred, vb)
                (loss_p + loss_v).backward()
                self.opt.step()

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

class MCTSNode: 
    __slots__ = ("state", "parent", "children", "visit_count", "total_value", "prior", "move")
    
    """
    Monte carlo tree search data structure

    The point here is that each node representrs a game state
    
    """
    def __init__(self, state: Engine.GameState, parent: 'MCTSNode|None' = None,
                 prior: float = 0.0, move: Engine.move | None = None):
        self.state = state
        self.parent = parent
        self.children: dict[Engine.move, MCTSNode] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.move = move
    def q(self):
        return self.total_value / self.visit_count if self.visit_count else 0.0
    def is_leaf(self):
        return not self.children



class MCTS:
    def __init__(self, nn: NeuraNet, simulations: int = 200, c_puct: float = 1.4):
        self.nn = nn
        self.N = simulations
        self.c = c_puct

    def search(self, root_state: Engine.GameState) -> dict[Engine.move, float]:
        root = MCTSNode(root_state)
        # Expand root once
        policy, _ = self.nn.predict(root_state)
        for mv, p in policy.items():
            root.children[mv] = MCTSNode(self._next_state(root_state, mv), parent=root, prior=p, move=mv)

        for _ in range(self.N):
            leaf = self._select(root)
            value = self._expand_and_eval(leaf)
            self._backprop(leaf, value)

        visits = np.array([ch.visit_count for ch in root.children.values()], dtype=np.float32)
        if visits.sum() == 0:
            return {}
        probs = visits / visits.sum()
        return dict(zip(root.children.keys(), probs))

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_leaf():
            best, best_score = None, -1e9
            sqrt_visits = np.sqrt(node.visit_count)
            for ch in node.children.values():
                u = self.c * ch.prior * sqrt_visits / (1 + ch.visit_count)
                score = ch.q() + u
                if score > best_score:
                    best, best_score = ch, score
            node = best
        return node

    def _expand_and_eval(self, node: MCTSNode) -> float:
        term = self._terminal_value(node.state)
        if term is not None:
            return term
        policy, value = self.nn.predict(node.state)
        for mv, p in policy.items():
            node.children[mv] = MCTSNode(self._next_state(node.state, mv), parent=node, prior=p, move=mv)
        return value

    def _backprop(self, node: MCTSNode, value: float):
        while node:
            node.visit_count += 1
            node.total_value += value
            value = -value
            node = node.parent

    @staticmethod
    def _next_state(state: Engine.GameState, mv: Engine.move) -> Engine.GameState:
        nxt = copy.deepcopy(state)
        nxt.make_move(mv)
        return nxt

    @staticmethod
    def _terminal_value(state: Engine.GameState):
        if state.check_mate:
            return 1 if not state.white_to_move else -1
        if state.stale_mate:
            return 0
        return None


 
def self_play(nn: NeuraNet, sims: int = 200) -> list[tuple[np.ndarray, dict[Engine.move, float], int]]:
    
    """
    play a game by using the self play with MCTS moves. 
    Each MCTS output (state, move prob) is saved for the training

    alg: 
    - run mcts to get move probs 
    - select a move 
    - save the state and move prob for trainging
    
    """
    game = Engine.GameState()
    mcts = MCTS(nn, sims)
    history = []
    while not (game.check_mate or game.stale_mate):
        probs = mcts.search(game)
        if not probs:
            break
        history.append((game.get_current_state(), probs, None))
        mv = random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
        game.make_move(mv)
    outcome = 0
    if game.check_mate:
        outcome = 1 if not game.white_to_move else -1
    return [(b, p, outcome) for (b, p, _) in history]


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(13, 128, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.flat  = nn.Flatten()
        self.fc    = nn.Linear(256 * 8 * 8, 1024)
        self.policy_head = nn.Linear(1024, 8**4)
        self.value_head  = nn.Linear(1024, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.flat(x)
        x = F.relu(self.fc(x))
        return self.policy_head(x), torch.tanh(self.value_head(x)).squeeze(1)

class PolicyNN(nn.Module): 
    pass 

if __name__ == "__main__": 
    net = NeuraNet()
    samples = self_play(net, sims=30)
    print(f"Self‑play generated {len(samples)} positions → training…")
    net.train(samples, epochs=1)
    net.save_model("chess_model.pth")