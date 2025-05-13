import numpy as np
import random 
import Engine 
import copy 
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


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


# AlphaMC_complete_with_policy.py
# Augmented with a **heuristic policy network** to avoid purely random play in
# the very first self‑play games.  The small net is cheap to evaluate and gives
# MCTS a sensible prior before the big CNN has learned anything.

from __future__ import annotations

import copy
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import Engine

NUM_MOVES = 8 ** 4  

def move_to_index(mv: Engine.move) -> int:
    return ((mv.start_row * 8 + mv.start_col) * 64) + (mv.end_row * 8 + mv.end_col)

#   HeuristicPolicy – 
class HeuristicPolicy:
    """first test implementation of a rule based scorer, the probability distribution over legal moves.
    """

    _piece_value = {"p": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}
    _centre = {(3, 3), (3, 4), (4, 3), (4, 4)}

    def _material_score(self, board):
        s = 0
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece != "--":
                    val = self._piece_value[piece[1]]
                    s += val if piece[0] == "w" else -val
        return s

    def predict(self, state: Engine.GameState) -> Dict[Engine.move, float]:
        moves = state.get_all_valid_moves()
        if not moves:
            return {}

        base_score = self._material_score(state.board)
        scores: List[float] = []
        for mv in moves:
            next_state = copy.deepcopy(state)
            next_state.make_move(mv)
            sc = self._material_score(next_state.board) - base_score
            # bonuses
            if mv.is_pawn_promotion:
                sc += 0.9
            if (mv.end_row, mv.end_col) in self._centre:
                sc += 0.1
            scores.append(sc)

        # Convert to positive logits
        scores_np = np.array(scores, dtype=np.float32)
        min_s = scores_np.min()
        scores_np -= min_s  # shift
        if scores_np.sum() == 0:
            probs = np.full_like(scores_np, 1 / len(moves))
        else:
            probs = scores_np / scores_np.sum()
        return dict(zip(moves, probs.tolist()))

#   CNN (policy + value) 
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
        self.policy_head = nn.Linear(1024, NUM_MOVES)
        self.value_head  = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.flat(x)
        x = F.relu(self.fc(x))
        return self.policy_head(x), torch.tanh(self.value_head(x)).squeeze(1)

class NeuraNet:
    def __init__(self, device: str | None = None, beta: float = 0.7):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = ChessNet().to(self.device)
        self.opt    = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()
        self.hp = HeuristicPolicy()
        self.beta = beta  # weight for blending heuristic policy (0..1)

    # inference
    @torch.no_grad()
    def predict(self, state: Engine.GameState) -> Tuple[Dict[Engine.move, float], float]:
        board = torch.tensor(state.get_current_state(), dtype=torch.float32, device=self.device).unsqueeze(0)
        logits_cnn, value = self.model(board)
        probs_cnn = torch.softmax(logits_cnn, dim=1).squeeze(0).cpu().numpy()

        legal_moves = state.get_all_valid_moves()
        if not legal_moves:
            return {}, 0.0

        # Heuristic prior
        probs_heur = self.hp.predict(state)

        move_probs: Dict[Engine.move, float] = {}
        total = 0.0
        for mv in legal_moves:
            p_cnn  = probs_cnn[move_to_index(mv)]
            p_heur = probs_heur.get(mv, 0.0)
            p = (1 - self.beta) * p_cnn + self.beta * p_heur
            move_probs[mv] = p
            total += p
        if total == 0:
            move_probs = {m: 1/len(legal_moves) for m in legal_moves}
        else:
            move_probs = {m: p/total for m, p in move_probs.items()}
        return move_probs, float(value.item())

    # training 
    def train(self, samples: List[Tuple[np.ndarray, Dict[Engine.move, float], int]],
              epochs: int = 5, batch_size: int = 32):
        if not samples:
            return
        X , P , V = [], [], []
        for board, pi, outcome in samples:
            X.append(torch.tensor(board, dtype=torch.float32))
            vec = np.zeros(NUM_MOVES, dtype=np.float32)
            for mv, p in pi.items():
                vec[move_to_index(mv)] = p
            P.append(torch.tensor(vec, dtype=torch.float32))
            V.append(float(outcome))
        X = torch.stack(X).to(self.device)
        P = torch.stack(P).to(self.device)
        V = torch.tensor(V, dtype=torch.float32).to(self.device)

        for _ in range(epochs):
            perm = torch.randperm(len(samples), device=self.device)
            X, P, V = X[perm], P[perm], V[perm]
            for i in range(0, len(samples), batch_size):
                xb, pb, vb = X[i:i+batch_size], P[i:i+batch_size], V[i:i+batch_size]
                self.opt.zero_grad()
                logits, v_pred = self.model(xb)
                loss_p = self.kl(F.log_softmax(logits, dim=1), pb)
                loss_v = self.mse(v_pred, vb)
                (loss_p + loss_v).backward()
                self.opt.step()

    # persistence 
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

# ────────────────────────────────────────────────────────────────────────────────
class MCTSNode:
    __slots__ = ("state", "parent", "children", "visit_count", "total_value", "prior", "move")
    def __init__(self, state: Engine.GameState, parent: Optional['MCTSNode'] = None,
                 prior: float = 0.0, move: Optional[Engine.move] = None):
        self.state = state
        self.parent = parent
        self.children: Dict[Engine.move, 'MCTSNode'] = {}
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

    def search(self, root_state: Engine.GameState) -> Dict[Engine.move, float]:
        root = MCTSNode(root_state)
        # first expansion uses blended priors from nn.predict
        priors, _ = self.nn.predict(root_state)
        for mv, p in priors.items():
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
        priors, value = self.nn.predict(node.state)
        for mv, p in priors.items():
            node.children[mv] = MCTSNode(self._next_state(node.state, mv), parent=node, prior=p, move=mv)
        return value

    def _backprop(self, node: MCTSNode, value: float):
        while node:
            node.visit_count += 1
            node.total_value += value
            value = -value
            node = node.parent

    @staticmethod
    def _next_state(state: Engine.GameState, mv: Engine.move):
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

# ────────────────────────────────────────────────────────────────────────────────

def self_play(nn: NeuraNet, sims: int = 300) -> List[Tuple[np.ndarray, Dict[Engine.move, float], int]]:
    game = Engine.GameState()
    mcts = MCTS(nn, simulations=sims)
    history = []
    while not (game.check_mate or game.stale_mate):
        probs = mcts.search(game)
        if not probs:
            break
        history.append((game.get_current_state(), probs, None))
        mv = max(probs.items(), key=lambda kv: kv[1])[0]  # pick best move (no randomness)
        game.make_move(mv)
    outcome = 0
    if game.check_mate:
        outcome = 1 if not game.white_to_move else -1
    return [(b, p, outcome) for (b, p, _) in history]

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    net = NeuraNet()
    positions = self_play(net, sims=60)
    print(f"Generated {len(positions)} positions → training 1 epoch…")
    net.train(positions, epochs=1)
    net.save_model("chess_model.pth")
