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

    def move_to_str(self, move):
        """
        Helper function that converts a Move object's list (move.l) into a string.
        Format: "row1,col1 row2,col2 ..."
        """
        return ' '.join([f"{pos[0]},{pos[1]}" for pos in move.l])

    def get_move(self, move):
        # If an opponent move is provided, update our board accordingly.
        if len(move) != 0:
            # Convert the received move to a string and then back to a Move object via from_str
            opponent_move_str = self.move_to_str(move)
            opponent_move_obj = Move.from_str(opponent_move_str)
            print("Opponent move:", opponent_move_obj.l)
            self.board.make_move(move, self.opponent[self.color])
        else:
            # No move provided means we are the first player.
            self.color = 1

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
                # In the simulated board, our move was just made,
                # so we pass self.color as the last mover.
                move_value = self.minimax(simulated_board, self.max_depth - 1, -float('inf'), float('inf'), False,
                                          self.color)
                if move_value > best_value:
                    best_value = move_value
                    best_move = move_option

        # Fallback: If no move was selected (should not happen), choose the first available move.
        if best_move is None:
            best_move = movesets[0][0]

        # Print our selected move using from_str
        my_move_str = self.move_to_str(best_move)
        my_move_obj = Move.from_str(my_move_str)
        print("My move:", my_move_obj.l)
        # Update our board with our chosen move.
        self.board.make_move(best_move, self.color)
        return best_move

    def minimax(self, board, depth, alpha, beta, maximizingPlayer, last_mover):
        """
        Minimax algorithm with Alpha-Beta Pruning.
        The parameter 'last_mover' indicates the color of the player who made the last move.
        """
        # Check for terminal state using the last mover.
        winner = board.is_win(last_mover)
        if depth == 0 or winner != 0:
            return self.evaluate_board(board)

        if maximizingPlayer:
            maxEval = -float('inf')
            currentColor = self.color
            movesets = board.get_all_possible_moves(currentColor)
            # If no moves are available, treat as a losing position.
            if not movesets:
                return -float('inf')
            for moves in movesets:
                for move_option in moves:
                    simulated_board = copy.deepcopy(board)
                    simulated_board.make_move(move_option, currentColor)
                    # After currentColor makes a move, update last_mover.
                    eval = self.minimax(simulated_board, depth - 1, alpha, beta, False, currentColor)
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
                    # After the opponent makes a move, update last_mover.
                    eval = self.minimax(simulated_board, depth - 1, alpha, beta, True, currentColor)
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
        # Check win conditions for both players.
        winner_our = board.is_win(self.color)
        winner_opp = board.is_win(self.opponent[self.color])
        if winner_our == self.color:
            return float('inf')
        elif winner_opp == self.opponent[self.color]:
            return -float('inf')

        my_pieces = 0
        opp_pieces = 0

        # Attempt to count pieces from the board's internal representation.
        try:
            for row in board.board:
                for piece in row:
                    if piece is not None:
                        if piece.get_color() == self.color:
                            my_pieces += 1
                        else:
                            opp_pieces += 1
        except AttributeError:
            # Fallback: Use mobility if the board layout is unavailable.
            my_moves = sum(len(moves) for moves in board.get_all_possible_moves(self.color))
            opp_moves = sum(len(moves) for moves in board.get_all_possible_moves(self.opponent[self.color]))
            return my_moves - opp_moves

        # Material score: weight each piece (e.g., 10 points each).
        material_score = (my_pieces - opp_pieces) * 10

        # Mobility score: difference in number of available moves.
        my_moves = sum(len(moves) for moves in board.get_all_possible_moves(self.color))
        opp_moves = sum(len(moves) for moves in board.get_all_possible_moves(self.opponent[self.color]))
        mobility_score = my_moves - opp_moves

        return material_score + mobility_score