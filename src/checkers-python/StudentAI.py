import random
import math
import copy
import sys
from BoardClasses import Board, Move
from concurrent.futures import ThreadPoolExecutor

class MCTSNode:
    def __init__(self, board, parent=None, last_move=None, color_to_move=1):
        """MCTS tree node with board state and UCB1 tracking."""
        self.board = board
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.wins = 0
        self.untried_moves = []
        self.color_to_move = color_to_move
        self.last_move = last_move

    def is_terminal_node(self):
        """Check if the game is over."""
        return self.board.is_win(1) != 0 or self.board.is_win(2) != 0

    def is_fully_expanded(self):
        """Check if all moves are explored."""
        return len(self.untried_moves) == 0

    def q(self):
        """Average win rate."""
        return float(self.wins) / self.visits if self.visits > 0 else 0

    def ucb1(self, exploration_constant):
        """Upper Confidence Bound for Trees (UCT)"""
        if self.visits == 0:
            return float('inf')
        return self.q() + exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)


class StudentAI:
    def __init__(self, col, row, p):
        """MCTS AI with defensive heuristic strategy."""
        print("DEBUG: StudentAI with Heuristic-based MCTS", file=sys.stderr)
        self.col = col
        self.row = row
        self.p = p
        self.board = Board(col, row, p)
        self.board.initialize_game()

        self.first_turn = True
        self.color = None
        self.opponent = {1: 2, 2: 1}

        # MCTS Parameters
        self.num_simulations = 3000
        self.exploration_param = math.sqrt(2)
        self.max_rollout_depth = 1000

    def _get_all_moves(self, board, color):
        """ Returns all possible moves for a given player, prioritizing captures. """
        movesets = board.get_all_possible_moves(color)
        capture_moves = []
        normal_moves = []

        for group in movesets:
            for move in group:
                if len(move.seq) > 1:  # Capture moves have longer sequences
                    capture_moves.append(move)
                else:
                    normal_moves.append(move)

        # Always prioritize capture moves if available
        return capture_moves if capture_moves else normal_moves

    def get_move(self, move):
        """Main function to select the best move using MCTS with heuristic rollouts."""
        if self.first_turn:
            self.first_turn = False
            self.color = 1 if len(move) == 0 else 2

        if len(move) != 0:
            self.board.make_move(move, self.opponent[self.color])

        movesets = self.board.get_all_possible_moves(self.color)
        if not movesets:
            return Move([])

        root = MCTSNode(copy.deepcopy(self.board), color_to_move=self.color)
        root.untried_moves = self._get_all_moves(root.board, root.color_to_move)

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not node.is_terminal_node() and node.untried_moves:
                node = self._expand(node)
            result = self._simulate(node)
            self._backpropagate(node, result)

        best_child = self._best_child(root, exploration=0)
        best_move = best_child.last_move
        self.board.make_move(best_move, self.color)
        return best_move
    
    def _backpropagate(self, node, result):
        """Backpropagate simulation results up the tree to update win/loss statistics."""
        while node is not None:
            node.visits += 1
            node.wins += result  # Reward for winning (+1), losing (-1), or draw (0)
            result = -result  # Flip result since opponent benefits from my loss
            node = node.parent

    def _best_child(self, node, exploration):
        """Select the best child node using UCB1 (Upper Confidence Bound)."""
        if not node.children:
            return None  # Fail-safe, should not happen in a valid game

        return max(
            node.children.values(),
            key=lambda child: child.ucb1(exploration) if exploration > 0 else child.q()
        )

    def _select(self, node):
        """Select the best child using UCB1 and heuristic safety checks."""
        while not node.is_terminal_node() and node.is_fully_expanded():
            best_child = self._best_child(node, self.exploration_param)

            # Defensive pruning: avoid moves that allow immediate capture
            if self._is_immediate_capture(best_child.board, self.opponent[best_child.color_to_move]):
                break
            node = best_child
        return node

    def _expand(self, node):
        """Expand a new child node, prioritizing king and capture moves."""
        if not node.untried_moves:
            return node

        capture_moves = [move for move in node.untried_moves if len(move.seq) > 1]
        king_moves = [move for move in node.untried_moves if move.seq[-1][0] in {0, self.row - 1}]

        move = capture_moves.pop() if capture_moves else king_moves.pop() if king_moves else node.untried_moves.pop()
        node.untried_moves.remove(move)

        new_board = copy.deepcopy(node.board)
        new_board.make_move(move, node.color_to_move)

        next_color = self.opponent[node.color_to_move]
        child_node = MCTSNode(new_board, node, move, next_color)
        child_node.untried_moves = self._get_all_moves(new_board, next_color)
        node.children[move] = child_node
        return child_node

    def _simulate(self, node):
        """Run a rollout with heuristic priorities: kinging, safety, and forced capture avoidance."""
        board_copy = copy.deepcopy(node.board)
        color_to_move = node.color_to_move
        depth = 0

        while depth < self.max_rollout_depth:
            if board_copy.is_win(self.color) == self.color:
                return +1.0
            elif board_copy.is_win(self.opponent[self.color]) == self.opponent[self.color]:
                return -1.0

            movesets = board_copy.get_all_possible_moves(color_to_move)
            if not movesets:
                return -1.0 if color_to_move == self.color else +1.0

            depth += 1
            all_moves = [m for group in movesets for m in group]
            move = self._prioritize_safe_moves(all_moves, board_copy, color_to_move)
            board_copy.make_move(move, color_to_move)
            color_to_move = self.opponent[color_to_move]

        return self._heuristic_evaluation(board_copy)
    
    def _is_future_capture_threat(self, board, move, color):
        """Check if this move will place the piece in a position to be captured in the opponent's next move."""
        simulated_board = copy.deepcopy(board)
        simulated_board.make_move(move, color)

        opponent_color = self.opponent[color]
        opponent_moves = simulated_board.get_all_possible_moves(opponent_color)

        for move_group in opponent_moves:
            for opponent_move in move_group:
                if move.seq[-1] in opponent_move.seq:  # If my final position is in their attack path
                    return True  # This is dangerous

                # Check if the opponent can chain multiple captures after taking this piece
                simulated_board_after_capture = copy.deepcopy(simulated_board)
                simulated_board_after_capture.make_move(opponent_move, opponent_color)
                next_moves = simulated_board_after_capture.get_all_possible_moves(opponent_color)

                for next_move_group in next_moves:
                    for next_move in next_move_group:
                        if move.seq[-1] in next_move.seq:  # My move results in a capture chain
                            return True

        return False  # Safe move

    
    def _is_trap_setup(self, board, move, color):
        """
        Check if making this move sets up the opponent for a strong counterattack,
        such as a multi-jump or easy king promotion.
        """
        simulated_board = copy.deepcopy(board)

        possible_moves = [m for group in simulated_board.get_all_possible_moves(color) for m in group]
        if move not in possible_moves:
            return False  # Ignore this check if the move is invalid

        simulated_board.make_move(move, color)  

        opponent_color = self.opponent[color]
        opponent_moves = simulated_board.get_all_possible_moves(opponent_color)

        for move_group in opponent_moves:
            for opponent_move in move_group:
                if len(opponent_move.seq) > 1:  # Multi-capture setup
                    return True
                if opponent_move.seq[-1][0] in {0, self.row - 1}:  # Easy king promotion
                    return True

        return False

    def _prioritize_safe_moves(self, moves, board, color):
        """Choose the best move prioritizing safety, king promotions, and running away from danger."""
        
        safe_moves = []
        escape_moves = []
        safe_captures = []
        safe_king_moves = []
        edge_moves = []
        neutral_moves = []

        possible_moves = [m for group in board.get_all_possible_moves(color) for m in group]  # Get all valid moves

        for move in moves:
            if move not in possible_moves:  # Ignore invalid moves
                continue

            simulated_board = copy.deepcopy(board)
            simulated_board.make_move(move, color)

            # **New: Detect Future Capture Threats**
            if self._is_future_capture_threat(board, move, color):
                escape_moves.append(move)  # Avoid this move at all costs

            # Standard Safety Checks
            if self._is_immediate_capture(simulated_board, self.opponent[color]) or \
            self._is_trap_setup(simulated_board, move, color):
                continue  # Avoid these moves entirely

            safe_moves.append(move)

            # Prioritize captures that don't expose us
            if len(move.seq) > 1:
                safe_captures.append(move)

            # Prioritize kinging when safe
            if move.seq[-1][0] in {0, self.row - 1}:
                safe_king_moves.append(move)

            # Prioritize staying on the edge if it avoids opponent attacks
            if move.seq[-1][1] in {0, self.col - 1}:
                edge_moves.append(move)

            # If a move is neutral (not immediately advantageous but not bad), store it
            neutral_moves.append(move)

        # **🔥 New: Prioritize ESCAPE moves first**
        if escape_moves:  
            return random.choice(escape_moves)
        if safe_captures:
            return random.choice(safe_captures)
        if safe_king_moves:
            return random.choice(safe_king_moves)
        if edge_moves:
            return random.choice(edge_moves)
        if safe_moves:
            return random.choice(safe_moves)
        if neutral_moves:
            return random.choice(neutral_moves)

        return random.choice(possible_moves)  # Default to any move if none are safe

    def _is_immediate_capture(self, board, opponent_color):
        """Check if a move results in an immediate capture."""
        opponent_moves = board.get_all_possible_moves(opponent_color)
        return any(len(move.seq) > 1 for group in opponent_moves for move in group)

    def _heuristic_evaluation(self, board):
        """Evaluate board position considering piece safety, king advantage, and board control."""
        
        piece_weight = 1
        king_weight = 4  # Increase king value
        edge_bonus = 3  # Apply conditionally
        safety_penalty = -2
        king_proximity_bonus = 0.5  # Encourage pieces moving forward safely

        my_score = 0
        opp_score = 0

        for r in range(self.row):
            for c in range(self.col):
                piece = board.board[r][c]
                if piece is None:  # **Fix: Ensure we're only processing valid pieces**
                    continue

                color = piece.get_color()
                if color not in self.opponent:  # **Fix: Skip invalid pieces**
                    continue  

                value = king_weight if piece.is_king else piece_weight

                # Give edge bonus only if it's actually useful
                if c in {0, self.col - 1} and r not in {0, self.row - 1}:  # Avoid already kinged pieces
                    value += edge_bonus

                # Encourage gradual movement toward king row
                if not piece.is_king:
                    if color == 1:
                        value += king_proximity_bonus * (r / self.row)
                    else:
                        value += king_proximity_bonus * ((self.row - r) / self.row)

                # Apply penalty if the piece is in an immediately capturable position
                if (r, c) in self._threatened_positions(board, self.opponent[color]):
                    value += safety_penalty

                if color == self.color:
                    my_score += value
                else:
                    opp_score += value

        return my_score - opp_score

    def _threatened_positions(self, board, opponent_color):
        """Find threatened positions by the opponent."""
        return {(move.seq[-1][0], move.seq[-1][1]) for group in board.get_all_possible_moves(opponent_color) for move in group}
