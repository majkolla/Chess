import pygame as p
import Engine
import os
import chessAI

## board 

wdith = height = 500
dimension = 8 #8x8 che ss dimension 
size = height // dimension
max_fps = 10
images = {}

##                  loading image
def load_images(): 
    pieces = ["wp", "wR", "wN", "wB", "wQ", "wK", "bp", "bR", "bN", "bB", "bQ", "bK"]
    for piece in pieces: # C:\Users\micha\OneDrive\Desktop\mainKod\Python\chess\pieces
        images[piece] = p.transform.scale(p.image.load(f'{piece}.png'), (size, size))  # Change absolute search to the pictures 

#looping in all of the images and saving them. Also could be accessed by calling "image["xx"]"
 
#Main driver for the code. This will handle input and updadting graphics accordingly

def main():
    p.init()
    screen = p.display.set_mode((wdith, height))
    clock = p.time.Clock()
    screen.fill(p.Color("white"))
    gs = Engine.GameState()
    valid_moves = gs.get_all_valid_moves() 
    move_made = False 
    animate = False #flag variable for when we sould animate stuff 
    load_images() 
    running = True
    sq_selected = () #keep track of the last square selected, will be a tuple and have x and y coordinates 
    player_clicks = [] # two tuples, keeps track of the 2 clicks 
    playerOne = False #True if human white
    playerTwo = False #False if AI black 
    game_over = False
    while running:
        humanTurn = (gs.white_to_move and playerOne) or (not gs.white_to_move and playerTwo)
        for e in p.event.get(): 
            if e.type == p.QUIT:
                running = False
    #MOUSE MOVEMENT!
            elif e.type == p.MOUSEBUTTONDOWN:
                if humanTurn and not game_over: 
                    location = p.mouse.get_pos() #(x,y) for the mouse
                    col = location[0]//size
                    row = location[1]//size
                    if sq_selected == (row, col):
                        sq_selected = ()   # unselect if the player select the same square twice
                        player_clicks = []
                    else:
                        sq_selected = (row, col)
                        player_clicks.append(sq_selected)
                    if len(player_clicks) == 2:
                        move = Engine.move(player_clicks[0], player_clicks[1], gs.board)
                        print(move.get_chess_notation())
                        for i in range(len(valid_moves)):
                            if move == valid_moves[i]:
                                gs.make_move(valid_moves[i])
                                move_made = True
                                animate = False
                                sq_selected = ()    #reset 
                                player_clicks = []  #reset
                        if not move_made:
                            player_clicks = [sq_selected]
        #KEYPRESSES! 
            elif e.type == p.KEYDOWN:
                if e.key == p.K_z:
                    gs.undo_move()
                    move_made = True
                    animate = False
                if e.key == p.K_r: #reset the board
                    gs = Engine.GameState()
                    valid_moves = gs.get_all_valid_moves()
                    sq_selected = ()
                    player_clicks = []
                    move_made = False
                    animate = False


        
        # AI MOVE

        if not humanTurn and not game_over:
            AIMove = chessAI.findBestMove(gs, valid_moves)
            if AIMove is None:
                AIMove = chessAI.find_random_move(valid_moves)
            gs.make_move(AIMove)
            move_made = True
            animate = True

        if move_made:
            if animate: 
                animate_move(gs.move_log[-1], screen, gs.board,clock)
            valid_moves = gs.get_all_valid_moves()
            move_made = False
            animate = False
        
        
        draw_game_state(screen, gs, valid_moves, sq_selected)
        
        if gs.check_mate:
            game_over = True
            if gs.white_to_move:
                drawText(screen, "Black wins by checkmate")
            else: 
                drawText(screen, "White wins by checkmate")
        elif gs.stale_mate:
            drawText(screen, "Stalemate")

        
        clock.tick(max_fps)
        p.display.flip()

""" UI stuff """
def Highlight_square(screen, gs, valid_moves, sq_selected):
    if sq_selected != (): 
        r,c = sq_selected
        if gs.board[r][c][0] == ("w" if gs.white_to_move else 'b'):
            s = p.Surface((size, size))
            s.set_alpha(100) #transperancy value 
            s.fill(p.Color("red"))
            screen.blit(s, (c*size, r*size))
            #highlight moves from that piece
            s.fill(p.Color('yellow'))
            for move in valid_moves: 
                if move.start_row == r and move.start_col == c:
                    screen.blit(s, (size * move.end_col,size* move.end_row))

# All of the graphics for the game 
def draw_game_state(screen, gs, valid_moves, sq_selected):
    draw_squares(screen) #draw squares on the board
    Highlight_square(screen, gs, valid_moves, sq_selected)
    draw_pieces(screen, gs.board) #draw pieces on top of the squares

# draw the squars on the board, top left square is always light 


def draw_squares(screen):
    global colors
    colors = [p.Color("white"), p.Color("darkgrey")]
    for i in range(dimension):
        for k in range(dimension):
            color = colors[((i+k) % 2)] 
            p.draw.rect(screen, color, p.Rect(k*size, i*size, size, size ))


# Draw the pieces on the board, therefore the board must be drawn first lol 

def draw_pieces(screen, board):
    for i in range(dimension):
        for k in range(dimension):
            piece = board[i][k]
            if piece != "--":
                screen.blit(images[piece], p.Rect(k*size, i*size, size, size))

def animate_move(move, screen, board, clock): 
    global colors
    coords = [] #list of coords that the anuimation will move
    dR = move.end_row - move.start_row
    dC = move.end_col - move.start_col
    frame_per_square = 10 #frames to move one square
    frame_count = (abs(dR) + abs(dC))  * frame_per_square
    for frame in range(frame_count + 1):
        r, c = (move.start_row + dR * frame/frame_count, move.start_col + dC * frame/frame_count)
        draw_squares(screen)
        draw_pieces(screen, board)
        #erase the piece from the ending square
        color = colors[(move.end_row + move.end_col) % 2]
        end_square = p.Rect(move.end_col* size, move.end_row * size, size, size)
        p.draw.rect(screen, color, end_square)
        if move.piece_captured != '--':
            screen.blit(images[move.piece_captured], end_square)
        #draw moving piece
        screen.blit(images[move.piece_moved], p.Rect(c*size, r*size, size, size))
        p.display.flip()
        clock.tick(60)

def drawText(screen, text):
    font = p.font.SysFont("Times new roman", 40, True, False)
    textObject = font.render(text,0, p.Color("Black"))
    textLocation = p.Rect(0,0,wdith, height).move(wdith/2 - textObject.get_width()/2, height/2 - textObject.get_height()/2)
    screen.blit(textObject, textLocation)

if __name__ == "__main__":
    main()

# %%






        