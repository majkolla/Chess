
'''
This class will be responsible for staring all the information of the current state of a
chess game, as well as determining the valid moves at the current state. 
'''
import numpy as np 

class GameState():
    # 2d 8x8 list dimensional board, each element has 2 characters 
    # the first represents color and the second represents the piece 
    def __init__(self):
        self.board = [
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bp", "bp", "bp", "bp", "bp", "bp", "bp", "bp"],
            ["--", "--", "--", "--", "--", "--", "--", "--"], # represent an empty space as string 
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["wp", "wp", "wp", "wp", "wp", "wp", "wp", "wp"],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]]
        self.white_to_move = True
        self.move_log = []
        self.move_functions = {'p': self.get_pawn_moves, 'R': self.get_rock_moves, 'N':self.get_Knight_moves, 
                               'B':self.get_Bishop_moves, 'Q':self.get_Queen_moves, 'K': self.get_King_moves}
        
        self.white_king_location = (7,4)
        self.black_king_location = (0,4)
        self.check_mate = False 
        self.stale_mate = False 
        self.enpassant_moves = ()
        self.current_Castling_rights = CastleingRights(True, True, True, True)
        self.castle_rightslog = [CastleingRights(self.current_Castling_rights.wks, self.current_Castling_rights.wqs, 
                                                self.current_Castling_rights.bks, self.current_Castling_rights.bqs )]
    def get_current_state(self): 
        """ I want it to return a 3D tensor of the state       
        
        """


        state_tensor = np.zeros((12, 8, 8), dtype=np.int8)

        piece_encoder : dict =  {
        "wp": 0, "wN": 1, "wB": 2, "wR": 3, "wQ": 4, "wK": 5,
        "bp": 6, "bN": 7, "bB": 8, "bR": 9, "bQ": 10, "bK": 11
    }
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece != "--":
                    channel = piece_encoder[piece]
                    state_tensor[channel, r, c] = 1
        side_to_move = np.full((1, 8, 8), 1 if self.white_to_move else 0, dtype=np.int8)
        full_state_tensor = np.concatenate([state_tensor, side_to_move], axis=0)
        return full_state_tensor

    def make_move(self, move):
        self.board[move.start_row][move.start_col] = "--"
        self.board[move.end_row][move.end_col] = move.piece_moved
        self.move_log.append(move)
        self.white_to_move = not self.white_to_move
        if move.piece_moved == "wK":
            self.white_king_location = (move.end_row, move.end_col)
        elif move.piece_moved == "bK":
            self.black_king_location = (move.end_row, move.end_col)
        
        # get pawn promotion 
        if move.is_pawn_promotion:  
            self.board[move.end_row][move.end_col] = move.piece_moved[0] + 'Q'

        #Enpassant 
        if move.enpassant_move:
            self.board[move.start_row][move.end_col] = '--'
        
        if (move.piece_moved[1] == 'p') and abs(move.start_row - move.end_row) == 2: #when a pawn makes to-step move 
            self.enpassant_moves = ((move.start_row + move.end_row)//2, move.start_col)
        else:
            self.enpassant_moves = ()
        #castling move 

        if move.isCastleMove:
            if move.end_col - move.start_col == 2: #ks castle 
                self.board[move.end_row][move.end_col-1] = self.board[move.end_row][move.end_col+1]
                self.board[move.end_row][move.end_col+1] = '--'
            else: #qs castle 
                self.board[move.end_row][move.end_col+1] = self.board[move.end_row][move.end_col-2]
                self.board[move.end_row][move.end_col-2] = '--'

        #Update castling rights when a king or a rook moves
        self.update_castle_rights(move)
        self.castle_rightslog.append(CastleingRights(self.current_Castling_rights.wks, self.current_Castling_rights.wqs, 
                                                self.current_Castling_rights.bks, self.current_Castling_rights.bqs))

    def undo_move(self): 
        if len(self.move_log) != 0:
            move = self.move_log.pop()
            self.board[move.start_row][move.start_col] = move.piece_moved
            self.board[move.end_row][move.end_col] = move.piece_captured
            self.white_to_move = not self.white_to_move
            if move.piece_moved == "wK":
                self.white_king_location = (move.start_row, move.start_col)
            elif move.piece_moved == "bK":
                self.black_king_location = (move.start_row, move.start_col)
            #undo enpassant 
            if move.enpassant_move:
                self.board[move.end_row][move.end_col] = "--"
                self.board[move.start_row][move.end_col] = move.piece_captured
                self.enpassant_moves = (move.end_row, move.end_col)
            if move.piece_moved[1] == 'p' and abs(move.start_row - move.end_row) == 2:
                self.enpassant_moves = ()

            #undo castling rights 
            self.castle_rightslog.pop()
            new_rights = self.castle_rightslog[-1]
            self.current_Castling_rights = CastleingRights(new_rights.wks, new_rights.wqs, new_rights.bks, new_rights.bqs)
            #undo castling move 
            if move.isCastleMove:
                if move.end_col - move.start_col == 2: #ks
                    self.board[move.end_row][move.end_col+1] = self.board[move.end_row][move.end_col-1]
                    self.board[move.end_row][move.end_col-1] = '--'
                else: #qs
                    self.board[move.end_row][move.end_col-2] = self.board[move.end_row][move.end_col+1]
                    self.board[move.end_row][move.end_col+1] = '--'
    
    def update_castle_rights(self, move):
        if move.piece_moved == "wK":
            self.current_Castling_rights.wks = False
            self.current_Castling_rights.wqs = False
        elif move.piece_moved == "bK":
            self.current_Castling_rights.bks = False
            self.current_Castling_rights.bqs = False
        elif move.piece_moved == "wR":
            if move.start_row == 7:
                if move.start_col == 0:
                    self.current_Castling_rights.wqs = False
                elif move.start_col == 7: #right rook 
                    self.current_Castling_rights.wks = False
        elif move.piece_moved == "bR":
            if move.start_row == 0:
                if move.start_col == 0:
                    self.current_Castling_rights.bqs = False
                elif move.start_col == 7: #right rook 
                    self.current_Castling_rights.bks = False       

    def get_all_moves(self):
        moves = []
        for r in range(len(self.board)):
            for c in range(len(self.board[r])):
                turn = self.board[r][c][0]
                if (turn == 'w' and self.white_to_move) or (turn == 'b' and not self.white_to_move):
                    piece = self.board[r][c][1]
                    self.move_functions[piece](r,c, moves)
        return moves

    '''
    these methods will return a list of all of the valid moves that every piece can make, without taking
    inte account for en passant, promotion, castling, and legal moves. 
    '''
    def get_pawn_moves(self, r, c, moves):
        if self.white_to_move: 
            if self.board[r-1][c] == "--": 
                moves.append(move( (r, c), (r-1, c), (self.board)))
                if r == 6 and self.board[r-2][c] == "--":
                    moves.append(move( (r, c), (r-2, c), self.board)) 
            if c+1<=7:
                if self.board[r-1][c+1][0] == 'b':
                    moves.append(move( (r, c), (r-1, c+1), self.board))
                elif (r-1, c+1) == self.enpassant_moves:
                    moves.append(move( (r, c), (r-1, c+1), self.board, is_enpassant_move=True))
            if c-1>=0:
                if self.board[r-1][c-1][0] == 'b':
                    moves.append(move( (r, c), (r-1, c-1), self.board))
                elif (r-1, c-1) == self.enpassant_moves:
                    moves.append(move( (r, c), (r-1, c-1), self.board, is_enpassant_move=True))
        else:
            if self.board[r+1][c]== "--":
                moves.append(move ((r,c), (r+1, c), self.board))
                if r == 1 and self.board[r+2][c] == "--":
                    moves.append(move( (r,c), (r+2,c), self.board))
            if c+1<=7:
                if self.board[r+1][c+1][0] == 'w':
                    moves.append(move( (r,c), (r+1, c+1), self.board))
                elif (r+1, c+1) == self.enpassant_moves:
                    moves.append(move( (r, c), (r+1, c+1), self.board, is_enpassant_move=True))
            if c-1>=0:
                if self.board[r+1][c-1][0] == 'w':
                    moves.append(move( (r,c), (r+1, c-1), self.board))
                elif (r+1, c-1) == self.enpassant_moves:
                    moves.append(move( (r, c), (r+1, c-1), self.board, is_enpassant_move=True))

    def get_rock_moves(self, r, c, moves):
        directions = ((-1, 0), (0, -1), (1, 0), (0,1))
        enemy_color = 'b' if self.white_to_move else 'w'  
        for d in directions:
            for i in range(1,8):
                end_row = r + d[0] * i
                end_col = c + d[1] * i
                if 0 <= end_row < 8 and 0 <= end_col < 8:
                    end_piece = self.board[end_row][end_col]
                    if end_piece == "--":
                        moves.append(move((r,c), (end_row, end_col), self.board))
                    elif end_piece[0] == enemy_color:
                        moves.append(move((r,c), (end_row, end_col), self.board ))
                        break
                    else: #friendly fire 
                        break
                else: #off board
                    break

    def get_Bishop_moves(self, r, c, moves):
        directions = ((1,1), (-1,-1), (1,-1), (-1,1))
        enemy_color = 'b' if self.white_to_move else 'w'
        for d in directions:
            for i in range(1,8):
                end_row = r + d[0] * i
                end_col = c + d[1] * i
                if 0 <= end_row < 8 and 0 <= end_col < 8:
                    end_piece = self.board[end_row][end_col]
                    if end_piece == "--":
                        moves.append(move((r,c), (end_row, end_col), self.board))
                    elif end_piece[0] == enemy_color:
                        moves.append(move((r,c), (end_row, end_col), self.board ))
                        break
                    else: #friendly fire
                        break
                else: #off board
                    break

    def get_Knight_moves(self, r, c, moves):
        Knight_moves = ( (-2,1), (-2,-1), (2,-1), (2,1), (1,2), (1,-2), (-1,2), (-1,-2) )
        enemy_color = 'b' if self.white_to_move else 'w'
        for K in Knight_moves:
            end_row = r + K[0]
            end_col = c + K[1]
            if 0 <= end_row < 8 and 0 <= end_col < 8:
                end_piece = self.board[end_row][end_col]
                if end_piece == "--":
                    moves.append(move((r,c), (end_row, end_col), self.board))
                elif end_piece[0] == enemy_color:
                    moves.append(move((r,c), (end_row, end_col), self.board))

    def get_King_moves(self, r, c, moves):
        king_moves = ((1,1), (1,0), (1,-1), (-1,1), (-1,0), (-1,-1), (0,1), (0,-1))
        enemy_color = 'b' if self.white_to_move else 'w'
        for i in range(8):
            end_row = r + king_moves[i][0]
            end_col = c + king_moves[i][1]
            if 0 <= end_row <8 and 0<= end_col < 8:
                end_piece = self.board[end_row][end_col]
                if end_piece == "--":
                    moves.append(move ((r,c), (end_row, end_col), self.board))
                if end_piece[0] == enemy_color:
                    moves.append(move((r,c), (end_row, end_col), self.board))    
    
    def get_castle_moves(self, r,c,moves):
        if self.square_under_attack(r,c):
            return #cant castle
        if (self.white_to_move and self.current_Castling_rights.wks) or (not self.white_to_move and self.current_Castling_rights.bks):
            self.get_king_side_castlemove(r,c,moves)
        if (self.white_to_move and self.current_Castling_rights.wqs) or (not self.white_to_move and self.current_Castling_rights.bqs):
            self.get_queenside_castlemove(r,c,moves)

    def get_king_side_castlemove(self, r,c,moves):
        if self.board[r][c+1] == '--' and self.board[r][c+2] == '--':
            if not self.square_under_attack(r,c+1) and not self.square_under_attack(r,c+2):
                moves.append(move((r,c), (r, c+2), self.board, isCastleMove=True))

    def get_queenside_castlemove(self,r,c,moves): 
        if (self.board[r][c-1] == '--') and (self.board[r][c-2] == '--') and (self.board[r][c-3] == '--'):
            if not self.square_under_attack(r,c-1) and not self.square_under_attack(r,c-2):
                moves.append(move((r,c),(r,c-2), self.board, isCastleMove=True))

    def get_Queen_moves(self, r, c, moves):
        self.get_Bishop_moves(r, c, moves)
        self.get_rock_moves(r, c, moves)

    def get_all_valid_moves(self): #checks for checks
        enpassant_possible_temp = self.enpassant_moves
        tempCastleRight = CastleingRights(self.current_Castling_rights.wks, self.current_Castling_rights.wqs, 
                                        self.current_Castling_rights.bks, self.current_Castling_rights.bqs)
        moves = self.get_all_moves() # generate all the moves
    
        for i in range(len(moves)-1, -1, -1): # For each move, make the move 
            self.make_move(moves[i])
            #generate all opoenntn moves 
            # for each enemt move, see if it attacks king  
            self.white_to_move = not self.white_to_move
            if self.in_check():
                moves.remove(moves[i])  # if attacks king, not valid move
            self.white_to_move = not self.white_to_move
            self.undo_move()
            print(moves)
        if len(moves) == 0:
            if self.in_check():
                self.check_mate = True
            else: 
                self.stale_mate = True 
    
        if self.white_to_move:
            self.get_castle_moves(self.white_king_location[0],self.white_king_location[1], moves)
        else:
            self.get_castle_moves(self.black_king_location[0], self.black_king_location[1], moves)
        
        self.enpassant_moves = enpassant_possible_temp
        self.current_Castling_rights = tempCastleRight

        return moves

    def in_check(self):
        if self.white_to_move:
            return self.square_under_attack(self.white_king_location[0], self.white_king_location[1])
        else:
            return self.square_under_attack(self.black_king_location[0], self.black_king_location[1])
       
    def square_under_attack(self, r, c):
        self.white_to_move = not self.white_to_move
        opponent_moves = self.get_all_moves()
        self.white_to_move = not self.white_to_move
        for move in opponent_moves:
            if move.end_row == r and move.end_col == c: #this square is under attack
                return True 
        return False

class CastleingRights():
    def __init__(self, wks, wqs, bks, bqs):
        self.wks = wks
        self.wqs = wqs
        self.bks = bks
        self.bqs = bqs

class move():
    # a move in chess contains two square, the start and the end square
    '''
    what moves are happening, and keeping track of the different starting squares and endning squares.
    defining what it mean to have a pieces captured and keeping track of it using the class. 
    '''

    ranks_to_rows = {"1":7, "2":6, "3":5, "4":4, "5":3, "6":2, "7":1, "8":0}
    rows_to_ranks = {i: k for k,i in ranks_to_rows.items()}
    files_to_cols = {"a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7}
    cols_to_files = {i: k for k, i in files_to_cols.items()}

    def __init__(self, start_sq, end_sq, board, is_enpassant_move=False, isCastleMove = False):
        self.start_row = start_sq[0]        
        self.start_col = start_sq[1]
        self.end_row = end_sq[0]
        self.end_col = end_sq[1]
        self.piece_moved = board[self.start_row][self.start_col]
        self.piece_captured = board[self.end_row][self.end_col]
        #pawn promotion
        self.is_pawn_promotion = (self.piece_moved == "wp" and self.end_row == 0) or (self.piece_moved == "bp" and self.end_row == 7)
        # Enpassant 
        self.enpassant_move = is_enpassant_move 
        if self.enpassant_move:
            self.piece_captured = 'wp' if self.piece_moved == 'bp' else 'bp'
        self.moveID = self.start_row * 1000 + self.start_col * 100 + self.end_row * 10 + self.end_col # gives a specic ID, to every move 
        self.isCastleMove = isCastleMove
    def __eq__(self, other):
        if isinstance(other, move):
            return self.moveID == other.moveID
        return False

    def __hash__(self):
        return hash(self.moveID)
        
    def get_rank_file(self, r, c):
        return self.cols_to_files[c] + self.rows_to_ranks[r]
    
    def get_col_row(self,r,f):
        return self.files_to_cols[f] + self.rows_to_ranks[r]

    def get_chess_notation(self):
        return self.get_rank_file(self.start_row, self.start_col) + self.get_rank_file(self.end_row, self.end_col)
