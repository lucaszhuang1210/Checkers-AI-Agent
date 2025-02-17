from random import randint
import copy
from BoardClasses import Move
from BoardClasses import Board


# The following part should be completed by students.
# Students can modify anything except the class name and existing functions and variables.
class StudentAI():

    def __init__(self, col, row, p):
        self.col = col
        self.row = row
        self.p = p
        self.board = Board(col, row, p)
        self.board.initialize_game()
        self.color = ''  # Will be set based on turn
        self.opponent = {1: 2, 2: 1}
        self.color = 2  # Default assignment; if we move first, color will be updated to 1.
        self.max_depth = 3  # Depth limit for the minimax search

    def get_move(self, move):
        # If an opponent move is provided, update our board accordingly.
        if hasattr(move, 'col') and hasattr(move, 'row') and move.col == -1 and move.row == -1:
            # We're the first player.
            self.color = 1
        else:
            # Otherwise, update our board with the opponent's move.
            self.board.make_move(move, self.opponent[self.color])

        # Retrieve all possible moves for our color.
        movesets = self.board.get_all_possible_moves(self.color)
        if not movesets:
            # No available moves; return an empty move.
            return Move([])

        best_move = None
        best_value = -float('inf')
        # Loop over every move option (movesets is a list of lists of moves)
        for moves in movesets:
            for move_option in moves:
                # Create a deep copy to simulate the move.
                simulated_board = copy.deepcopy(self.board)
                simulated_board.make_move(move_option, self.color)
                # Run minimax on the simulated board.
                move_value = self.minimax(simulated_board, self.max_depth - 1, -float('inf'), float('inf'), False)
                if move_value > best_value:
                    best_value = move_value
                    best_move = move_option

        # Fallback: If no move was selected (should not happen), choose the first available move.
        if best_move is None:
            best_move = movesets[0][0]

        # Update our board with our chosen move.
        self.board.make_move(best_move, self.color)
        return best_move

    def minimax(self, board, depth, alpha, beta, maximizingPlayer):
        winner = board.is_win(self.color)
        # Terminal condition: depth limit reached or game over.
        if depth == 0 or winner != 0:
            return self.evaluate_board(board)

        if maximizingPlayer:
            maxEval = -float('inf')
            currentColor = self.color
            movesets = board.get_all_possible_moves(currentColor)
            # If no moves are available, treat this as a losing position.
            if not movesets:
                return -float('inf')
            for moves in movesets:
                for move_option in moves:
                    simulated_board = copy.deepcopy(board)
                    simulated_board.make_move(move_option, currentColor)
                    eval = self.minimax(simulated_board, depth - 1, alpha, beta, False)
                    maxEval = max(maxEval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break  # Beta cutoff
                if beta <= alpha:
                    break
            return maxEval
        else:
            minEval = float('inf')
            currentColor = self.opponent[self.color]
            movesets = board.get_all_possible_moves(currentColor)
            # If no moves are available for the opponent, it's a winning position.
            if not movesets:
                return float('inf')
            for moves in movesets:
                for move_option in moves:
                    simulated_board = copy.deepcopy(board)
                    simulated_board.make_move(move_option, currentColor)
                    eval = self.minimax(simulated_board, depth - 1, alpha, beta, True)
                    minEval = min(minEval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break  # Alpha cutoff
                if beta <= alpha:
                    break
            return minEval

    def evaluate_board(self, board):
        """
        Heuristic evaluation function that considers:
          - Terminal states: win/loss
          - Material advantage: difference in the number of pieces
          - Mobility: difference in the number of available moves
        """
        winner = board.is_win(self.color)
        if winner == self.color:
            return float('inf')
        elif winner == self.opponent[self.color]:
            return -float('inf')

        my_pieces = 0
        opp_pieces = 0

        # Try to count pieces using board.board (assumed to be a 2D list of pieces).
        try:
            for row in board.board:
                for piece in row:
                    if piece is not None:
                        if piece.get_color() == self.color:
                            my_pieces += 1
                        else:
                            opp_pieces += 1
        except AttributeError:
            # Fallback: Use mobility as a proxy if board.board isn't available.
            my_moves = sum(len(moves) for moves in board.get_all_possible_moves(self.color))
            opp_moves = sum(len(moves) for moves in board.get_all_possible_moves(self.opponent[self.color]))
            return my_moves - opp_moves

        # Material score: each piece is given a weight (e.g., 10 points per piece).
        material_score = (my_pieces - opp_pieces) * 10

        # Mobility score: the difference in the number of possible moves.
        my_moves = sum(len(moves) for moves in board.get_all_possible_moves(self.color))
        opp_moves = sum(len(moves) for moves in board.get_all_possible_moves(self.opponent[self.color]))
        mobility_score = my_moves - opp_moves

        return material_score + mobility_score