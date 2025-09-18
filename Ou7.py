#!/usr/bin/env python3
from __future__ import print_function
from collections import namedtuple, defaultdict
from itertools import count
import time, math
###############################################################################
# Piece-Square tables. These tables help the engine evaluate how good a position
# is based on where pieces are placed on the board. Higher values mean better
# positions for that piece.
###############################################################################

# Values for each piece type. The king is extremely valuable to prevent easy mates.
piece = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}

# Position tables for each piece, giving bonuses or penalties for specific squares.
pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}

# Pad the tables to fit the internal board representation (120 squares).
for k, table in pst.items():
    padrow = lambda row: (0,) + tuple(x + piece[k] for x in row) + (0,)
    pst[k] = sum((padrow(table[i * 8 : i * 8 + 8]) for i in range(8)), ())
    pst[k] = (0,) * 20 + pst[k] + (0,) * 20

###############################################################################
# Global constants used in the chess logic and search.
###############################################################################

# Internal board representation: 120 characters, with padding for edge detection.
A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    "         \n"  #   0 -  9
    "         \n"  #  10 - 19
    " rnbqkbnr\n"  #  20 - 29
    " pppppppp\n"  #  30 - 39
    " ........\n"  #  40 - 49
    " ........\n"  #  50 - 59
    " ........\n"  #  60 - 69
    " ........\n"  #  70 - 79
    " PPPPPPPP\n"  #  80 - 89
    " RNBQKBNR\n"  #  90 - 99
    "         \n"  # 100 -109
    "         \n"  # 110 -119
)

# Directions for piece movements: North, East, South, West and diagonals.
N, E, S, W = -10, 1, 10, -1
directions = {
    "P": (N, N+N, N+W, N+E),
    "N": (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    "B": (N+E, S+E, S+W, N+W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N+E, S+E, S+W, N+W),
    "K": (N, E, S, W, N+E, S+E, S+W, N+W)
}

# Mate values: Used to detect checkmates. High values beyond normal piece captures.
MATE_LOWER = piece["K"] - 10 * piece["Q"]
MATE_UPPER = piece["K"] + 10 * piece["Q"]

# Search tuning constants: Control quiescence search and evaluation.
QS = 40
QS_A = 140
EVAL_ROUGHNESS = 15

opt_ranges = dict(
    QS = (0, 300),
    QS_A = (0, 300),
    EVAL_ROUGHNESS = (0, 50),
)

###############################################################################
# Chess logic: Represents positions and moves.
###############################################################################

# A simple data structure for moves, including promotion piece if any.
Move = namedtuple("Move", "i j prom")

# Position class: Holds the board state, score, castling rights, en passant, etc.
class Position(namedtuple("Position", "board score wc bc ep kp")):
    """A state of a chess game
    board -- a 120 char representation of the board
    score -- the board evaluation
    wc -- the castling rights for white, [queen side, king side]
    bc -- the castling rights for black, [queen side, king side]
    ep - the en passant square
    kp - the king passant square (for castling detection)
    """

    # Generate all possible legal moves from this position.
    def gen_moves(self):
        # Loop through each square on the board.
        for i, p in enumerate(self.board):
            if not p.isupper():
                continue  # Skip if not our piece (uppercase means current player).
            # For each direction the piece can move.
            for d in directions[p]:
                # Step in that direction until blocked.
                for j in count(i + d, d):
                    q = self.board[j]
                    # Stop if off board or friendly piece.
                    if q.isspace() or q.isupper():
                        break
                    # Special rules for pawns: forward moves, captures, promotions.
                    if p == "P":
                        if d in (N, N + N) and q != ".": break
                        if d == N + N and (i < A1 + N or self.board[i + N] != "."): break
                        if (
                            d in (N + W, N + E)
                            and q == "."
                            and j not in (self.ep, self.kp, self.kp - 1, self.kp + 1)
                        ):
                            break
                        # Promotion on last rank.
                        if A8 <= j <= H8:
                            for prom in "NBRQ":
                                yield Move(i, j, prom)
                            break
                    # Yield the move.
                    yield Move(i, j, "")
                    # Stop for non-sliding pieces or after capture.
                    if p in "PNK" or q.islower():
                        break
                    # Handle castling by moving the rook.
                    if i == A1 and self.board[j + E] == "K" and self.wc[0]:
                        yield Move(j + E, j + W, "")
                    if i == H1 and self.board[j + W] == "K" and self.wc[1]:
                        yield Move(j + W, j + E, "")

    # Rotate the board for the opponent's perspective (swaps colors and reverses board).
    def rotate(self, nullmove=False):
        """Rotates the board, preserving enpassant, unless nullmove"""
        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119 - self.ep if self.ep and not nullmove else 0,
            119 - self.kp if self.kp and not nullmove else 0,
        )

    # Apply a move to create a new position.
    def move(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i + 1 :]
        # Copy and reset special squares.
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        # Make the move on the board.
        board = put(board, j, board[i])
        board = put(board, i, ".")
        # Update castling rights if rook or king moves/captures.
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        # Handle king castling.
        if p == "K":
            wc = (False, False)
            if abs(j - i) == 2:
                kp = (i + j) // 2
                board = put(board, A1 if j < i else H1, ".")
                board = put(board, kp, "R")
        # Handle pawn specials: promotion, double move, en passant.
        if p == "P":
            if A8 <= j <= H8:
                board = put(board, j, prom)
            if j - i == 2 * N:
                ep = i + N
            if j == self.ep:
                board = put(board, j + S, ".")
        # Rotate for next player.
        return Position(board, score, wc, bc, ep, kp).rotate()

    # Calculate the evaluation change from this move.
    def value(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        # Base on piece-square tables.
        score = pst[p][j] - pst[p][i]
        # Add for captures.
        if q.islower():
            score += pst[q.upper()][119 - j]
        # Detect attacks on king during castling.
        if abs(j - self.kp) < 2:
            score += pst["K"][119 - j]
        # Adjust for castling rook movement.
        if p == "K" and abs(i - j) == 2:
            score += pst["R"][(i + j) // 2]
            score -= pst["R"][A1 if j < i else H1]
        # Pawn specials.
        if p == "P":
            if A8 <= j <= H8:
                score += pst[prom][j] - pst["P"][j]
            if j == self.ep:
                score += pst["P"][119 - (j + S)]
        return score

###############################################################################
# Search logic: The engine's brain for finding the best move.
###############################################################################

# Transposition table entry for cached search results.
Entry = namedtuple("Entry", "lower upper")

# Searcher class: Handles the search algorithm.
class Searcher:
    def __init__(self):
        self.tp_score = {}  # Transposition table for scores.
        self.tp_move = {}  # Transposition table for best moves.
        self.history = set()  # To detect repetitions.
        self.nodes = 0  # Count nodes searched.

    # Bound the score of the position relative to gamma (MTD-bi search helper).
    def bound(self, pos, gamma, depth, can_null=True):
        """Returns a bound on the true score of the position."""
        self.nodes += 1
        depth = max(depth, 0)  # Treat negative depth as 0 for quiescence.

        # If the position is a loss (no king), return mate value.
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER

        # Check transposition table for previous results.
        entry = self.tp_score.get((pos, depth, can_null), Entry(-MATE_UPPER, MATE_UPPER))
        if entry.lower >= gamma: return entry.lower
        if entry.upper < gamma: return entry.upper

        # Avoid repetitions.
        if can_null and depth > 0 and pos in self.history:
            return 0

        # Generator for moves, including null moves and killers.
        def moves():
            # Null move pruning: Skip move if conditions met.
            if depth > 2 and can_null and abs(pos.score) < 500:
                yield None, -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3)

            # In quiescence search (depth 0), stand pat if no captures.
            if depth == 0:
                yield None, pos.score

            # Killer move: Best from previous search.
            killer = self.tp_move.get(pos)
            if not killer and depth > 2:
                self.bound(pos, gamma, depth - 3, can_null=False)
                killer = self.tp_move.get(pos)

            # Quiescence: Only high-value moves at depth 0.
            val_lower = QS - depth * QS_A
            if killer and pos.value(killer) >= val_lower:
                yield killer, -self.bound(pos.move(killer), 1 - gamma, depth - 1)

            # Sort and yield other moves.
            for val, move in sorted(((pos.value(m), m) for m in pos.gen_moves()), reverse=True):
                if val < val_lower:
                    break
                if depth <= 1 and pos.score + val < gamma:
                    yield move, pos.score + val if val < MATE_LOWER else MATE_UPPER
                    break
                yield move, -self.bound(pos.move(move), 1 - gamma, depth - 1)

        # Find the best score from moves.
        best = -MATE_UPPER
        for move, score in moves():
            best = max(best, score)
            if best >= gamma:
                if move is not None:
                    self.tp_move[pos] = move
                break

        # Handle stalemate/mate detection.
        if depth > 2 and best == -MATE_UPPER:
            flipped = pos.rotate(nullmove=True)
            in_check = self.bound(flipped, MATE_UPPER, 0) == MATE_UPPER
            best = -MATE_LOWER if in_check else 0

        # Update transposition table.
        if best >= gamma:
            self.tp_score[pos, depth, can_null] = Entry(best, entry.upper)
        if best < gamma:
            self.tp_score[pos, depth, can_null] = Entry(entry.lower, best)

        return best

    # Main search: Iterative deepening with MTD-bi.
    def search(self, history):
        """Iterative deepening MTD-bi search"""
        self.nodes = 0
        self.history = set(history)
        self.tp_score.clear()

        gamma = 0
        for depth in range(1, 1000):
            lower, upper = -MATE_LOWER, MATE_LOWER
            while lower < upper - EVAL_ROUGHNESS:
                score = self.bound(history[-1], gamma, depth, can_null=False)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
                yield depth, gamma, score, self.tp_move.get(history[-1])
                gamma = (lower + upper + 1) // 2

###############################################################################
# Helper functions for parsing and rendering moves in algebraic notation.
###############################################################################

# Parse algebraic notation (e.g., 'e2') to internal square index.
def parse(c):
    fil, rank = ord(c[0]) - ord("a"), int(c[1]) - 1
    return A1 + fil - 10 * rank

# Render internal square index to algebraic notation (e.g., 'e2').
def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord("a")) + str(-rank + 1)

###############################################################################
# Interactive mode: Replaces UCI with simple text-based play.
###############################################################################

# Start with initial position.
hist = [Position(initial, 0, (True, True), (True, True), 0, 0)]
searcher = Searcher()

# Greet and get user preferences.
print("Ou7 Interface")
think_time = float(input("Bot think time (In ms)? "))
think_time = think_time / 1000
side = input("Which side do you want to play? (white/black)? ").strip().lower()
user_white = side == 'white'

# Main game loop.
while True:
    pos = hist[-1]
    # Determine whose turn it is (white to move if odd history length).
    white_to_move = len(hist) % 2 == 1
    current_turn = 'white' if white_to_move else 'black'

    # Check for game over: No moves left.
    moves = list(pos.gen_moves())
    if not moves:
        flipped = pos.rotate(nullmove=True)
        in_check = searcher.bound(flipped, MATE_UPPER, 0) == MATE_UPPER
        if in_check:
            winner = 'Bot' if (user_white == white_to_move) else 'User'
            print(f"Checkmate! {winner} wins.")
        else:
            print("Stalemate!")
        break

    if (user_white and white_to_move) or (not user_white and not white_to_move):
        # User's turn: Get and validate move.
        while True:
            move_str = input("User Move: ").strip()
            if move_str.lower() in ['quit', 'resign']:
                print("Game ended by user.")
                exit(0)
            try:
                i = parse(move_str[:2])
                j = parse(move_str[2:4])
                prom = move_str[4:].upper() if len(move_str) > 4 else ''
                # Adjust coordinates if black to move (board is rotated).
                if not white_to_move:
                    i, j = 119 - i, 119 - j
                move = Move(i, j, prom)
                if move not in moves:
                    raise ValueError
                # Apply move.
                hist.append(pos.move(move))
                break
            except:
                print("Invalid move. Try again (e.g., e2e4 or e7e8q for promotion).")
    else:
        # Bot's turn: Search for move within time limit.
        start = time.time()
        best_move = None
        best_move_str = None
        for depth, gamma, score, move in searcher.search(hist):
            if score >= gamma and move:
                i, j = move.i, move.j
                # Adjust render if black to move.
                if not white_to_move:
                    i, j = 119 - i, 119 - j
                best_move_str = render(i) + render(j) + move.prom.lower()
                best_move = move
            if best_move_str and time.time() - start > think_time:
                break
        if best_move:
            hist.append(pos.move(best_move))
            print("Bot Move:", best_move_str)
        else:
            print("Bot resigns (no move found).")
            break
