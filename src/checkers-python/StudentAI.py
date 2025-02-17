from random import randint
import copy
from BoardClasses import Move
from BoardClasses import Board

# The following part should be completed by students.
# Students can modify anything except the class name
# and the existing function signatures and variable names.
class StudentAI():

    def __init__(self, col, row, p):
        self.col = col
        self.row = row
        self.p = p
        self.board = Board(col, row, p)
        self.board.initialize_game()
        # By default, assume we are player 2; if we move first, we'll switch to 1
        self.color = 2
        self.opponent = {1: 2, 2: 1}

        # You can adjust the search depth based on performance or board size
        self.max_depth = 3

    def get_move(self, move):
        """
        Called by the game engine. 'move' is the opponent's move if any. 
        We must return our own move as a Move object.
        """
        # If the opponent made a move, apply it to our Board state
        if len(move) != 0:
            self.board.make_move(move, self.opponent[self.color])
        else:
            # If 'move' is empty, we are going first (color = 1)
            self.color = 1

        # Collect all possible moves for our color
        possible_moves = self.board.get_all_possible_moves(self.color)
        if not possible_moves:
            # No moves available => must return None
            return None

        # Use Minimax + Alpha-Beta to pick the best move
        best_move = None
        best_value = float('-inf')

        # possible_moves is a list of lists (grouped by jump sequences),
        # so we iterate over each group and each Move within it
        for group_of_moves in possible_moves:
            for candidate_move in group_of_moves:
                # Simulate this candidate move
                board_copy = copy.deepcopy(self.board)
                board_copy.make_move(candidate_move, self.color)

                # Evaluate using minimax
                value = self.minimax(
                    board_copy,
                    depth=self.max_depth,
                    alpha=float('-inf'),
                    beta=float('inf'),
                    maximizing_player=False,
                    last_mover=self.color
                )

                if value > best_value:
                    best_value = value
                    best_move = candidate_move

        # If, for some reason, no best_move was chosen (unlikely),
        # just pick a random valid move to avoid crashing.
        if best_move is None:
            best_move = possible_moves[0][0]

        # Apply our chosen move to the real board
        self.board.make_move(best_move, self.color)
        return best_move

    def minimax(self, board, depth, alpha, beta, maximizing_player, last_mover):
        """
        Standard Minimax search with Alpha-Beta pruning.
        - 'board' is the game state to evaluate.
        - 'depth' is how many levels of recursion remain.
        - 'alpha' and 'beta' maintain the best/worst values for pruning.
        - 'maximizing_player' = True if it's our "AI" turn, False if opponent's turn.
        - 'last_mover' is the color (1 or 2) who made the previous move,
          so we can check if that move produced a winning state.
        """
        # First, check if the last move ended the game.
        winner = board.is_win(last_mover)
        if winner == self.color:
            return float('inf')   # We already won
        elif winner == self.opponent[self.color]:
            return float('-inf')  # We lost

        # If we ran out of search depth, do a static evaluation
        if depth == 0:
            return self.evaluate_board(board)

        if maximizing_player:
            # AI's turn to move
            best_val = float('-inf')
            all_moves = board.get_all_possible_moves(self.color)

            # If no moves, it's effectively a losing position
            if not all_moves:
                return float('-inf')

            for group_of_moves in all_moves:
                for move_option in group_of_moves:
                    new_board = copy.deepcopy(board)
                    new_board.make_move(move_option, self.color)

                    val = self.minimax(
                        new_board,
                        depth - 1,
                        alpha,
                        beta,
                        maximizing_player=False,
                        last_mover=self.color
                    )
                    best_val = max(best_val, val)
                    alpha = max(alpha, val)
                    if beta <= alpha:
                        # Alpha-Beta cutoff
                        break
                if beta <= alpha:
                    break
            return best_val
        else:
            # Opponent's turn to move
            best_val = float('inf')
            opp_color = self.opponent[self.color]
            all_moves = board.get_all_possible_moves(opp_color)

            # If opponent has no moves, that means we are effectively winning
            if not all_moves:
                return float('inf')

            for group_of_moves in all_moves:
                for move_option in group_of_moves:
                    new_board = copy.deepcopy(board)
                    new_board.make_move(move_option, opp_color)

                    val = self.minimax(
                        new_board,
                        depth - 1,
                        alpha,
                        beta,
                        maximizing_player=True,
                        last_mover=opp_color
                    )
                    best_val = min(best_val, val)
                    beta = min(beta, val)
                    if beta <= alpha:
                        # Alpha-Beta cutoff
                        break
                if beta <= alpha:
                    break
            return best_val

    def evaluate_board(self, board):
        """
        A simple heuristic:
          - If there's a winner, assign ±∞.
          - Otherwise, count difference in number of pieces (our pieces - opponent's).
            The greater, the better.
          - You can also add other heuristics (e.g. kings vs. non-kings, mobility, etc.).
        """
        # Check if there's a winner from the perspective of the last move
        # But typically we call .evaluate_board on a non-terminal state, so let's do a direct check:
        winner_our = board.is_win(self.color)
        if winner_our == self.color:
            return float('inf')
        winner_opp = board.is_win(self.opponent[self.color])
        if winner_opp == self.opponent[self.color]:
            return float('-inf')

        # Count pieces
        my_pieces = 0
        opp_pieces = 0

        # Attempt direct piece counting
        try:
            for row in board.board:
                for piece in row:
                    if piece is not None:
                        # If your code uses piece.get_color(), change accordingly
                        if piece.color == self.color:
                            my_pieces += 1
                        else:
                            opp_pieces += 1
        except AttributeError:
            # If direct piece reference doesn't work, fallback to mobility difference
            my_moves = board.get_all_possible_moves(self.color)
            opp_moves = board.get_all_possible_moves(self.opponent[self.color])
            return len(my_moves) - len(opp_moves)

        return my_pieces - opp_pieces
