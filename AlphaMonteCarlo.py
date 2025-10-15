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


NUM_MOVES = 8 ** 4  # 4 096 from–to square pairs

#   Move <‑‑> index helpers

def move_to_index(mv: Engine.move) -> int:
    return ((mv.start_row * 8 + mv.start_col) * 64) + (mv.end_row * 8 + mv.end_col)

class PolicyNetSmall(nn.Module):
    """~150 K parameters. Fast enough for use at every tree node."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(13, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.flat  = nn.Flatten()
        self.fc    = nn.Linear(64 * 8 * 8, NUM_MOVES)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flat(x)
        return self.fc(x)  # raw logits

class ValueNet(nn.Module):
    """Deeper network (~2 M params) used only at leaf nodes → slower but OK."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(13, 128, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.flat  = nn.Flatten()
        self.fc    = nn.Linear(256 * 8 * 8, 512)
        self.val   = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.flat(x)
        x = F.relu(self.fc(x))
        return torch.tanh(self.val(x)).squeeze(1)

# ────────────────────────────────────────────────────────────────────────────────
class DualNet:
    """Handles inference & joint training for PolicyNetSmall + ValueNet."""
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetSmall().to(self.device)
        self.value  = ValueNet().to(self.device)
        # one optimiser for both parameter sets keeps it simple
        self.opt = torch.optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=1e-3)
        self.kl  = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()

    @torch.no_grad()
    def predict(self, state: Engine.GameState) -> Tuple[Dict[Engine.move, float], float]:
        board = torch.tensor(state.get_current_state(), dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(board).squeeze(0)
        p_full = torch.softmax(logits, dim=0).cpu().numpy()
        v = float(self.value(board).item())

        legal_moves = state.get_all_valid_moves()
        if not legal_moves:
            return {}, v
        move_probs: Dict[Engine.move, float] = {}
        total = 0.0
        for mv in legal_moves:
            p = p_full[move_to_index(mv)]
            move_probs[mv] = p
            total += p
        if total == 0:
            move_probs = {m: 1/len(legal_moves) for m in legal_moves}
        else:
            move_probs = {m: p/total for m, p in move_probs.items()}
        return move_probs, v

    # ————————————————— joint training
    def train(self, batch: List[Tuple[np.ndarray, Dict[Engine.move, float], int]], epochs=1, batch_sz=64):
        if not batch:
            return
        X, P, V = [], [], []
        for board, pi, outcome in batch:
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
            perm = torch.randperm(len(batch), device=self.device)
            X, P, V = X[perm], P[perm], V[perm]
            for i in range(0, len(batch), batch_sz):
                xb, pb, vb = X[i:i+batch_sz], P[i:i+batch_sz], V[i:i+batch_sz]
                self.opt.zero_grad()
                pol_logits = self.policy(xb)
                val_pred   = self.value(xb)
                loss_p = self.kl(F.log_softmax(pol_logits, dim=1), pb)
                loss_v = self.mse(val_pred, vb)
                (loss_p + loss_v).backward()
                self.opt.step()

    # ————————————————— persistence
    def save(self, path_root="dualnet"):
        torch.save(self.policy.state_dict(), f"{path_root}_policy.pth")
        torch.save(self.value.state_dict(),  f"{path_root}_value.pth")

    def load(self, path_root="dualnet"):
        self.policy.load_state_dict(torch.load(f"{path_root}_policy.pth", map_location=self.device))
        self.value.load_state_dict(torch.load(f"{path_root}_value.pth",  map_location=self.device))
        self.policy.eval(); self.value.eval()

# ────────────────────────────────────────────────────────────────────────────────
#   MCTS (unchanged API, now uses DualNet)
# ────────────────────────────────────────────────────────────────────────────────
class MCTSNode:
    __slots__ = ("state","parent","children","visit_count","total_value","prior","move")
    def __init__(self, state: Engine.GameState, parent: Optional['MCTSNode']=None,
                 prior=0.0, move: Optional[Engine.move]=None):
        self.state = state; self.parent = parent; self.prior = prior; self.move = move
        self.children: Dict[Engine.move,'MCTSNode'] = {}
        self.visit_count = 0; self.total_value = 0.0
    def q(self):
        return self.total_value / self.visit_count if self.visit_count else 0.0
    def is_leaf(self):
        return not self.children

class MCTS:
    def __init__(self, net: DualNet, sims=200, c_puct=1.4):
        self.net, self.N, self.c = net, sims, c_puct
    def search(self, root_state: Engine.GameState) -> Dict[Engine.move,float]:
        root = MCTSNode(root_state)
        priors, _ = self.net.predict(root_state)
        for mv,p in priors.items():
            root.children[mv] = MCTSNode(self._next(root_state,mv), parent=root, prior=p, move=mv)
        for _ in range(self.N):
            leaf = self._select(root)
            value = self._expand_eval(leaf)
            self._backprop(leaf, value)
        visits = np.array([ch.visit_count for ch in root.children.values()], dtype=np.float32)
        if visits.sum()==0: return {}
        probs = visits/visits.sum()
        return dict(zip(root.children.keys(), probs))
    def _select(self,n):
        while not n.is_leaf():
            best, best_s = None, -1e9; sqrt_p = np.sqrt(n.visit_count)
            for ch in n.children.values():
                u = self.c*ch.prior*sqrt_p/(1+ch.visit_count); s = ch.q()+u
                if s>best_s: best,best_s=ch,s
            n = best
        return n
    def _expand_eval(self,node):
        t = self._term(node.state)
        if t is not None: return t
        priors,val = self.net.predict(node.state)
        for mv,p in priors.items():
            node.children[mv] = MCTSNode(self._next(node.state,mv), parent=node, prior=p, move=mv)
        return val
    def _backprop(self,node,val):
        while node:
            node.visit_count+=1; node.total_value+=val; val=-val; node=node.parent
    @staticmethod
    def _next(state,mv):
        s = copy.deepcopy(state); s.make_move(mv); return s
    @staticmethod
    def _term(state):
        if state.check_mate: return 1 if not state.white_to_move else -1
        if state.stale_mate: return 0
        return None

# ────────────────────────────────────────────────────────────────────────────────
#   Self‑play generator
# ────────────────────────────────────────────────────────────────────────────────

def self_play(net: DualNet, sims=300) -> List[Tuple[np.ndarray, Dict[Engine.move,float], int]]:
    gs=Engine.GameState(); tree=MCTS(net,sims)
    hist=[]
    while not (gs.check_mate or gs.stale_mate):
        pi = tree.search(gs)
        if not pi: break
        hist.append((gs.get_current_state(), pi, None))
        mv = max(pi,itemgetter(1))[0]
        gs.make_move(mv)
    z = 0
    if gs.check_mate: z = 1 if not gs.white_to_move else -1
    return [(b,p,z) for (b,p,_) in hist]

if __name__ == "__main__":
    dual = DualNet()
    data = self_play(dual, sims=60)
    print("Positions:", len(data))
    dual.train(data, epochs=1)
    dual.save()
