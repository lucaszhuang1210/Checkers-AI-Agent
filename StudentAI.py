from operator import attrgetter
from random import randint
from BoardClasses import Move
from BoardClasses import Board
# The following part should be completed by students.
# Students can modify anything except the class name and exisiting functions and varibles.

from copy import deepcopy
import copy
import sys
import random
import math
import time

def get_opponent_color(color):
    return 2 if color == 1 else 1

def generate_random_move(board, color):
    """
    Given a board state and color, returns a random move.
    """
    moves = [move for sublist in board.get_all_possible_moves(color) for move in sublist]
    return random.choice(moves) if moves else None

class MCTSNode():
    def __init__(self, board, color_to_move=1, last_move=None, parent=None):
        self.board = deepcopy(board)
        self.color_to_move = color_to_move
        self.last_move = last_move
        self.parent = parent
        self.visits = 1
        self.wins_for_parent = 0
        self.ucb_value = 0

        # Execute nodes' first move
        if last_move is not None:
            self.board.make_move(last_move, get_opponent_color(self.color_to_move))

        # Expand possible moves at node creation
        self.children = {}
        if self.board.is_win(get_opponent_color(self.color_to_move)) == 0:
            moves = self.board.get_all_possible_moves(self.color_to_move)
            for move_set in moves:
                for move in move_set:
                    self.children[move] = None  # Children will be assigned upon expansion

    def backpropagate(self, win_for_parent):
        """
        Recursively update visit counts and win statistics up the tree.
        1 = win for parent, -1 = loss for parent, 0 = draw.
        """
        self.visits += 1

        # Update parent node
        if self.parent:
            self.parent.backpropagate(-win_for_parent)

        if win_for_parent > 0:
            self.wins_for_parent += win_for_parent
        elif win_for_parent == 0:
            self.wins_for_parent += 0.5  # Tie

        # Recalculate UCB value
        if self.parent:
            self.ucb_value = self.q() + math.sqrt(2) * math.sqrt(math.log(self.parent.visits) / self.visits)

    def q(self):
        """Return the average payoff (wins / visits) from AI's perspective."""
        return self.wins_for_parent / self.visits if self.visits > 0 else 0

class MCTS():
    def __init__(self, root):
        self.root = root

    def search(self, time_limit):
        """
        Perform Monte Carlo Tree Search until the time limit is reached.
        Returns the best move from the root node.
        """
        end_time = time_limit + time.time()

        while time.time() < end_time:
            node = self.selection(self.root)  # Selection step
            win_for_parent = self.simulate(node)  # Simulation step
            node.backpropagate(win_for_parent)  # Backpropagation step

        return self.best_move()  # Choose the best move

    def selection(self, node):
        """
        Recursively select the best child (highest UCB1) until a node that
        is either terminal or not fully expanded is found.
        """
        if not node.children:
            return node

        if None not in node.children.values():
            return self.selection(max(node.children.values(), key=attrgetter('ucb_value')))

        for move, child in node.children.items():
            if child is None:
                node.children[move] = MCTSNode(node.board, get_opponent_color(node.color), move, node)
                return self.expand(node)

    def expand(self, node):
        """
        Expands the first available unvisited child node.
        """
        for move, child in node.children.items():
            if child is None:
                node.children[move] = MCTSNode(node.board, get_opponent_color(node.color), move, node)
                return node.children[move]

    def simulate(self, node):
        """
        Simulates a random game from the given node and returns the outcome.
        """
        temp_board = deepcopy(node.board)
        temp_color = node.color
        win_flag = temp_board.is_win(get_opponent_color(temp_color))

        while not win_flag:
            temp_board.make_move(generate_random_move(temp_board, temp_color), temp_color)
            win_flag = temp_board.is_win(temp_color)
            temp_color = get_opponent_color(temp_color)

        if win_flag == get_opponent_color(node.color):
            return 1  # Win for parent
        elif win_flag == node.color:
            return -1  # Loss for parent
        return 0  # Draw

    def best_move(self):
        sorted_moves = sorted(self.root.children.items(), key=lambda x: x[1].visit_count, reverse=True)
        return sorted_moves[0][0]

class StudentAI:
    def __init__(self, col, row, p):
        print("DEBUG: StudentAI using Monte Carlo Tree Search.", file=sys.stderr)
        self.col = col
        self.row = row
        self.p = p

        self.board = Board(col, row, p)
        self.board.initialize_game()

        self.color = 2
        self.opponent = {1: 2, 2: 1}

        self.max_rollout_depth = MAX_ROLLOUT_DEPTH
        self.exploration_param = 1.41

    def get_move(self, move):
        # If opponent made a move, update our board.
        if move:
            self.board.make_move(move, self.opponent[self.color])
        else:
            # No move implies we are starting the game.
            self.color = 1

        # Get all possible moves for our current board.
        movesets = self.board.get_all_possible_moves(self.color)
        if not movesets:
            return Move([])  # No moves available

        # Create a root node representing the current board state.
        root = MCTSNode(
            board=copy.deepcopy(self.board),
            parent=None,
            last_move=None,
            color_to_move=self.color
        )
        # Initialize untried moves at the root.
        root.untried_moves = self._get_all_moves(root.board, root.color_to_move)

        # Create an MCTS instance with the current root.
        # Here we use a time-based search (e.g., 1.0 second) to run the simulations.
        mcts = MCTS(root, self.color, self.opponent,
                    exploration_param=self.exploration_param,
                    max_rollout_depth=self.max_rollout_depth)
        best_move = mcts.search(time_limit=1.0)

        # Make the chosen move on our board.
        self.board.make_move(best_move, self.color)
        return best_move

    # Helper function to collect and shuffle all moves from available movesets.
    def _get_all_moves(self, board, color):
        movesets = board.get_all_possible_moves(color)
        all_moves = [move for group in movesets for move in group]
        random.shuffle(all_moves)
        return all_moves

    # -------------------------------------------------------------------------
    # MCTS Core Functions
    # -------------------------------------------------------------------------


