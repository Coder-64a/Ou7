import chess, random, time, math, sys
board = chess.Board()
print(board)
while not board.is_game_over():
    while board.turn == chess.WHITE:
        try:
            user_move = input("Enter a move (like e2e4): ")
            user_mv = chess.Move.from_uci(user_move)
            if board.is_legal(user_mv):
                board.push(user_mv)
                print(board)
            else:
                print("Illegal move, try again.")
                continue
            break
        except:
            print("Invalid move format, try again.")
    while board.turn == chess.BLACK:
        try:
            user_move2 = input("Enter a move (like e2e4): ")
            user_mv2 = chess.Move.from_uci(user_move2)
            if board.is_legal(user_mv2):
                board.push(user_mv2)
                print(board)
            else:
                print("Illegal move, try again.")
                continue
            break
        except:
            print("Invalid move format, try again.")
    
