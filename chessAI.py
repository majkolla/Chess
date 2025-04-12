import random



""" Optimera detta skit program, i wanna kill myself
    If not optimize:
        You're a worthless piece of shit    """

piece_score = {"K": 0, "Q": 10, "R":5, "B": 3, "N": 3, "p":1}
Checkmate = 1000
Stalemate = 0
DEPTH = 2


def find_random_move(valid_moves):  #Random move function 
    return valid_moves[random.randint(0, len(valid_moves) -1)]

def find_best_move1(gs, valid_moves): #Material oriented function 
    turn_multiplier = 1 if gs.white_to_move else -1 
    opponent_MIN_maxscore = Checkmate
    bestPlayerMove = None
    random.shuffle(valid_moves)
    for playerMove in valid_moves: 
        gs.make_move(playerMove)
        opponent_moves = gs.get_all_valid_moves()
        random.shuffle(valid_moves)
        opponentmaxscore = -Checkmate
        for opponent_moves in opponent_moves:
            gs.make_move(opponent_moves)
            random.shuffle(valid_moves)
            if gs.check_mate:
                score = -turn_multiplier * Checkmate
            elif gs.stale_mate:
                score = Stalemate
            else:
                score = -turn_multiplier * score_material(gs.board)
            if score > opponentmaxscore: #if there is no capturing the score will be worth 0 and thereby it will take the first score = 0 as the best move
                opponentmaxscore = score 
            gs.undo_move()
        if opponent_MIN_maxscore > opponentmaxscore:
            opponent_MIN_maxscore = opponentmaxscore
            bestPlayerMove = playerMove
        gs.undo_move()
    return bestPlayerMove

def findBestMove(gs, valid_moves):
    global nextMove
    nextMove = None
    #findMoveMegaMaxAlphaBeta(gs, valid_moves, DEPTH,-Checkmate, Checkmate, 1 if gs.white_to_move else -1)
    #findMoveMegaMax(gs, valid_moves, DEPTH, 1 if gs.white_to_move else -1)
    find_best_move1(gs, valid_moves)
    #find_random_move(valid_moves)
    return nextMove

def find_move_minmax(gs, valid_moves, depth, white_to_move):
    global nextMove
    if depth == 0: 
        return score_material(gs.board)
    
    if white_to_move:
        maxScore = -Checkmate
        for move in valid_moves:
            gs.make_move(move)
            nextMoves = gs.get_all_valid_moves()
            score = find_move_minmax(gs, nextMoves, depth -1, False)
            if score > maxScore:
                maxScore = score
                if depth == DEPTH:
                    nextMove = move 
            gs.undo_move()
        return maxScore
    else:
        minScore = Checkmate
        for move in valid_moves:
            gs.make_move(move)
            nextMoves = gs.get_all_valid_moves()
            score = find_move_minmax(gs, nextMoves, depth -1, True)
            if score < minScore:
                minScore = score
                if depth == DEPTH:
                    nextMove = move 
            gs.undo_move()
        return minScore
         
def findMoveMegaMax(gs, validMoves, depth, TurnMultiplier): #find the max and (*-1) if blacks turn 
    global nextMove
    random.shuffle(validMoves)
    if depth == 0: 
        return TurnMultiplier * score_board(gs) 
    maxScore = - Checkmate
    for move in validMoves:
        gs.make_move(move)
        nextMoves = gs.get_all_valid_moves()
        score = -findMoveMegaMax(gs, nextMoves, depth-1, -TurnMultiplier)
        if score > maxScore:
            maxScore = score
            if depth == DEPTH:
                nextMove = move
        gs.undo_move()
    return maxScore

def findMoveMegaMaxAlphaBeta(gs, validMoves, depth, alpha, beta, TurnMultiplier):
    global nextMove
    random.shuffle(validMoves)

    if depth == 0: 
        return TurnMultiplier * score_board(gs)

    #move ordering to optimize  
    maxScore = - Checkmate
    for move in validMoves:
        random.shuffle(validMoves)
        gs.make_move(move)
        nextMoves = gs.get_all_valid_moves()
        score = -findMoveMegaMaxAlphaBeta(gs, nextMoves, depth-1,-beta, -alpha, -TurnMultiplier)
        if score > maxScore:
            maxScore = score
            if depth == DEPTH:
                nextMove = move
        gs.undo_move()
        if maxScore > alpha: #pruning 
            alpha = maxScore
        
        if alpha >= beta:
            break 

    return maxScore


def score_board(gs):
    if gs.check_mate:
        if gs.white_to_move:
            return - Checkmate
        else:
            return Checkmate
    elif gs.stale_mate:
        return Stalemate 
    score = 0
    for row in gs.board: 
        for square in row: 
            if square[0] == 'w':
                score += piece_score[square[1]]
            elif square[0] == 'b':
                score -= piece_score[square[1]]
    return score

def score_material(board):
    score = 0
    for row in board: 
        for square in row: 
            if square[0] == 'w':
                score += piece_score[square[1]]
            elif square[0] == 'b':
                score -= piece_score[square[1]]
    return score
