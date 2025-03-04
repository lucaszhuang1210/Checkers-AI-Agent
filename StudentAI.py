import random
import math
import copy
import sys
from BoardClasses import Board, Move


class MCTSNode:
    def __init__(self, board, parent=None, last_move=None, color_to_move=1):
        """
        MCTS tree node.
        - board: Current Board state
        - parent: Parent node in the search tree
        - children: Dict[move -> MCTSNode]
        - visits: Number of times this node was visited
        - wins: Summed result (e.g., +1 for wins, -1 for losses)
        - untried_moves: List of legal moves not yet expanded at this node
        - color_to_move: Which player's move it is in this node's state
        - last_move: The move that led from the parent to this node
        """
        self.board = board
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.wins = 0
        self.untried_moves = []
        self.color_to_move = color_to_move
        self.last_move = last_move

    def is_terminal_node(self):
        """
        Check if the game is over at this node (win or no moves left).
        """
        if self.board.is_win(1) != 0 or self.board.is_win(2) != 0:
            return True
        possible_moves = self.board.get_all_possible_moves(self.color_to_move)
        return (len(possible_moves) == 0)

    def is_fully_expanded(self):
        """
        A node is 'fully expanded' if all possible child moves have been explored.
        """
        return len(self.untried_moves) == 0

    def q(self):
        """
        Average result from this node's perspective (wins / visits).
        If visits=0, return 0 to avoid division by zero.
        """
        if self.visits == 0:
            return 0
        return float(self.wins) / float(self.visits)

    def ucb1(self, exploration_constant):
        """
        Upper Confidence Bound for Trees (UCT/ UCB1).
        """
        if self.visits == 0:
            return float('inf')  # Encourage exploring unvisited nodes
        return self.q() + exploration_constant * math.sqrt(
            math.log(self.parent.visits) / float(self.visits)
        )


class StudentAI:
    def __init__(self, col, row, p):
        """
        An AI that uses Monte Carlo Tree Search (MCTS) for decision-making,
        combined with a more advanced heuristic evaluation in rollouts.

        - Player 1 = Black, on top rows, moves downward
        - Player 2 = White, on bottom rows, moves upward
        """
        print("DEBUG: StudentAI with MCTS + improved heuristic", file=sys.stderr)
        self.col = col
        self.row = row
        self.p = p
        self.board = Board(col, row, p)
        self.board.initialize_game()

        # We don't know if we're player 1 (Black) or player 2 (White) until the first turn.
        # We'll detect it in get_move via whether move is empty or not.
        self.first_turn = True
        self.color = None  # Will set this properly on the first turn
        self.opponent = {1: 2, 2: 1}

        # MCTS Parameters
        self.num_simulations = 600
        self.exploration_param = math.sqrt(2)
        self.max_rollout_depth = 50

    def get_move(self, move):
        """
        The main function called each turn. Returns our chosen Move.

        If move == [], it means we are the first player to move.
        Otherwise, 'move' is the move the opponent just made.
        """
        # If this is our first time here, determine which color we are.
        if self.first_turn:
            self.first_turn = False
            if len(move) == 0:
                # We are the first player => Black = 1
                self.color = 1
            else:
                # We are the second player => White = 2
                self.color = 2

        # Apply the opponent's move if it's not empty
        if len(move) != 0:
            self.board.make_move(move, self.opponent[self.color])

        # Get all moves we can make now.
        movesets = self.board.get_all_possible_moves(self.color)
        if not movesets:
            # No moves available => must return an empty move => we lose
            return Move([])

        # Build the root node for MCTS
        root = MCTSNode(copy.deepcopy(self.board),
                        parent=None,
                        last_move=None,
                        color_to_move=self.color)
        root.untried_moves = self._get_all_moves(root.board, root.color_to_move)

        # Perform MCTS simulations
        for _ in range(self.num_simulations):
            node = self._select(root)
            if (not node.is_terminal_node()) and node.untried_moves:
                node = self._expand(node)
            result = self._simulate(node)
            self._backpropagate(node, result)

        # Best child (exploration=0 => purely exploit best Q value)
        best_child = self._best_child(root, exploration=0)
        best_move = best_child.last_move

        # Make that move on our real board
        self.board.make_move(best_move, self.color)
        return best_move

    def _select(self, node):
        """
        Descend through the tree until we find a node that either
          - is terminal, or
          - is not fully expanded.
        """
        while not node.is_terminal_node() and node.is_fully_expanded():
            node = self._best_child(node, self.exploration_param)
        return node

    def _expand(self, node):
        """
        Take one untried move from 'node', create a new child node for it,
        and return that child node.
        """
        move = node.untried_moves.pop()
        new_board = copy.deepcopy(node.board)
        new_board.make_move(move, node.color_to_move)

        next_color = self.opponent[node.color_to_move]
        child_node = MCTSNode(new_board, node, move, next_color)
        child_node.untried_moves = self._get_all_moves(new_board, next_color)
        node.children[move] = child_node
        return child_node

    def _simulate(self, node):
        """
        Simulate (rollout) from 'node' until we reach 'max_rollout_depth' or terminal.
        Return +1 if we eventually see a win for 'self.color',
               -1 if we see a loss for 'self.color',
               or a heuristic evaluation if we reach the depth limit.
        """
        board_copy = copy.deepcopy(node.board)
        color_to_move = node.color_to_move
        depth = 0

        while depth < self.max_rollout_depth:
            # Check if game ended:
            if board_copy.is_win(self.color) == self.color:
                return +1.0
            elif board_copy.is_win(self.opponent[self.color]) == self.opponent[self.color]:
                return -1.0

            movesets = board_copy.get_all_possible_moves(color_to_move)
            if not movesets:
                # No moves => this side loses
                return -1.0 if color_to_move == self.color else +1.0

            depth += 1
            # Randomly pick from all possible moves
            all_moves = [m for group in movesets for m in group]
            random_move = random.choice(all_moves)

            # Apply move
            board_copy.make_move(random_move, color_to_move)
            color_to_move = self.opponent[color_to_move]

        # If we exit the loop, we haven't reached a terminal state => use heuristic
        return self._heuristic_evaluation(board_copy)

    def _backpropagate(self, node, result):
        """
        Propagate the 'result' (e.g. +1 for win, -1 for loss) up the tree.
        """
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

    def _best_child(self, node, exploration):
        """
        Choose the child with the highest UCB1 if exploration != 0,
        or the highest average win rate (child.q()) if exploration=0.
        """
        if exploration == 0:
            # Pure exploitation
            return max(node.children.values(), key=lambda child: child.q())
        else:
            # UCB1 selection
            return max(node.children.values(), key=lambda child: child.ucb1(exploration))

    def _get_all_moves(self, board, color):
        """
        Flatten the list of lists returned by board.get_all_possible_moves(color)
        into a single list of Move objects.
        """
        movesets = board.get_all_possible_moves(color)
        return [m for group in movesets for m in group]

    def _heuristic_evaluation(self, board):
        """
        A more advanced evaluation function that considers:
          1) Piece count and king value
          2) Mobility (# of available moves)
          3) Proximity to promotion row for non-king pieces

        We assume:
          - color=1 (Black) is top->down
          - color=2 (White) is bottom->up
        """
        my_color = self.color
        opp_color = self.opponent[my_color]

        # 1) Piece/Kings scoring
        REGULAR_PIECE_VALUE = 1
        KING_VALUE = 3

        my_piece_score = 0
        opp_piece_score = 0

        # 2) Mobility scoring
        my_moves = board.get_all_possible_moves(my_color)
        opp_moves = board.get_all_possible_moves(opp_color)
        mobility_score = (len(my_moves) - len(opp_moves)) * 0.2

        # 3) Proximity to promotion
        PROXIMITY_BONUS = 0.1

        for r in range(self.row):
            for c in range(self.col):
                piece = board.board[r][c]
                if piece is not None:
                    color = piece.get_color()
                    value = KING_VALUE if piece.is_king else REGULAR_PIECE_VALUE

                    if color == my_color:
                        # Distances: Black=1 => (self.row-1 - r), White=2 => r
                        if not piece.is_king:
                            if color == 1:  # Black
                                distance_to_promotion = (self.row - 1 - r)
                            else:  # White
                                distance_to_promotion = r
                            rowBonus = (self.row - 1 - distance_to_promotion)
                            value += PROXIMITY_BONUS * rowBonus
                        my_piece_score += value
                    else:
                        # Opponent piece
                        if not piece.is_king:
                            if color == 1:  # Black
                                distance_to_promotion = (self.row - 1 - r)
                            else:  # White
                                distance_to_promotion = r
                            rowBonus = (self.row - 1 - distance_to_promotion)
                            value += PROXIMITY_BONUS * rowBonus
                        opp_piece_score += value

        # Final score: piece difference + mobility
        return (my_piece_score - opp_piece_score) + mobility_score
