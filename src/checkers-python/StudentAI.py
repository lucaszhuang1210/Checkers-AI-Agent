from random import randint
from BoardClasses import Move
from BoardClasses import Board
#The following part should be completed by students.
#Students can modify anything except the class name and exisiting functions and varibles.

import copy
import sys
import random

NUM_SIMULATION = 1000
MAX_ROLLOUT_DEPTH = 500

class MCTSNode:
    def __init__(self, board, parent=None, last_move=None, color_to_move=1):
        self.board = board
        self.parent = parent
        self.children = {}  # move -> MCTSNode
        self.visits = 0
        self.wins = 0
        self.untried_moves = []
        self.color_to_move = color_to_move
        self.last_move = last_move

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal_node(self):
        """Check if the game is finished at this node (win/lose/draw)."""
        # We can see if there's a winner for either color; 
        # if is_win(...) returns non-zero, the game is over.
        if self.board.is_win(1) != 0 or self.board.is_win(2) != 0:
            return True
        # Also, if no moves exist, it's effectively terminal for the current color
        movesets = self.board.get_all_possible_moves(self.color_to_move)
        return (len(movesets) == 0)

    def q(self):
        """Return the average payoff = wins / visits (from AI perspective)."""
        if self.visits == 0:
            return 0
        return float(self.wins) / float(self.visits)

    def ucb1(self, exploration_constant):
        """
        Return the UCB1 score:
          Q + c * sqrt( ln(parent.visits) / visits ).
        """
        if self.visits == 0:
            return float('inf')  # Encourage expansion of unvisited nodes
        return self.q() + exploration_constant * (
            (2 * (self.parent.visits))**0.5 / float(self.visits)
        )


class StudentAI():
    def __init__(self, col, row, p):
        """
        Initialize your AI with MCTS parameters. 
        'col', 'row', 'p' refer to the board dimensions and piece arrangement,
        just like in previous AIs.
        """
        print("DEBUG: StudentAI using Monte Carlo Tree Search.", file=sys.stderr)
        self.col = col
        self.row = row
        self.p = p

        self.board = Board(col, row, p)
        self.board.initialize_game()

        # By default, we'll assume we are color=2; if we see an empty move, we are color=1.
        self.color = 2
        self.opponent = {1: 2, 2: 1}

        # MCTS parameters
        self.num_simulations = NUM_SIMULATION    # The number of MCTS iterations per move. Increase if you have time
        self.max_rollout_depth = MAX_ROLLOUT_DEPTH  # Limit random rollout length (to avoid huge searching)

        self.exploration_param = 1.41 # Exploration constant 'c' in UCB1

    def get_move(self, move):
        """
        Called each turn by the game engine:
          - 'move' is the opponent's last move (or an empty Move if we move first).
          - Return a Move object as our chosen action.
        """
        # If the opponent just moved, apply it to our board.
        if len(move) != 0: 
            self.board.make_move(move, self.opponent[self.color])
        else:
            # If there's no opponent move, we move first => color=1
            self.color = 1

        # Check all possible moves. If none, return empty Move.
        movesets = self.board.get_all_possible_moves(self.color)
        if not movesets:
            return Move([])

        # Build a root node for MCTS from the current board state.
        root = MCTSNode(board=copy.deepcopy(self.board),
                        parent=None,
                        last_move=None,
                        color_to_move=self.color)

        # Initialize untried moves for the root
        root.untried_moves = self._get_all_moves(root.board, root.color_to_move)

        # Run MCTS for a fixed number of iterations
        for _ in range(self.num_simulations):
            # 1. Selection
            node = self._select(root)

            # 2. Expansion
            if (not node.is_terminal_node()) and (len(node.untried_moves) > 0):
                node = self._expand(node)

            # 3. Simulation (rollout)
            result = self._simulate(node)

            # 4. Backpropagation
            self._backpropagate(node, result)

        # After simulations, pick the move (child of root) with the highest visit count (or best average Q).
        best_child = self._best_child(root, 0)  # set exploration = 0 for final choice
        best_move = best_child.last_move  # The move that led to that child

        # Execute the best move on our real board
        self.board.make_move(best_move, self.color)
        return best_move

    # -------------------------------------------------------------------------
    # MCTS Core Functions
    # -------------------------------------------------------------------------
    def _select(self, node):
        """ 
        Selection: descend down the tree while the current node is fully expanded
        and not terminal, picking children via UCB1.
        """
        while (not node.is_terminal_node()) and node.is_fully_expanded():
            node = self._best_child(node, self.exploration_param)
        return node

    def _expand(self, node):
        """ 
        Expansion: create a new child for one of the node's untried moves.
        """
        move = node.untried_moves.pop()
        new_board = copy.deepcopy(node.board)
        new_board.make_move(move, node.color_to_move)

        # Next color to move
        next_color = self.opponent[node.color_to_move]

        child_node = MCTSNode(
            board=new_board,
            parent=node,
            last_move=move,
            color_to_move=next_color
        )
        # Prepare the child's untried moves
        child_node.untried_moves = self._get_all_moves(new_board, next_color)

        node.children[move] = child_node
        return child_node

    def _backpropagate(self, node, result):
        """
        Backpropagation: propagate the simulation result up the tree. 
        'result' is +1 if we eventually won, -1 if we lost, 0 if draw, from *our* perspective.
        """
        while node is not None:
            node.visits += 1
            # If we are the AI color= self.color, 
            # we add 'result' to node.wins if node's perspective is also self.color.
            # However, a simpler approach is to always store results from the 
            # perspective of self.color. So we just do:
            node.wins += result
            node = node.parent

    def _best_child(self, node, exploration):
        """
        Return the child with the highest UCB1 score if exploration>0 
        or the best average Q if exploration=0 (final move selection).
        """
        best = None
        best_score = float('-inf')
        for move, child in node.children.items():
            if exploration > 0:
                score = child.ucb1(exploration)
            else:
                # Final move selection => pick the highest average Q or highest visits
                score = child.q()

            if score > best_score:
                best_score = score
                best = child
        return best

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------
    def _get_all_moves(self, board, color):
        """
        Flatten all possible moves for 'color' from the board,
        because many checkers engines return a list-of-lists.
        """
        movesets = board.get_all_possible_moves(color)
        all_moves = []
        for group in movesets:
            for m in group:
                all_moves.append(m)
        return all_moves

    def _simulate(self, node):
        board_copy = copy.deepcopy(node.board)
        color_to_move = node.color_to_move
        rollout_depth = 0

        while True:
            # Provide the current board state and color to our move selector.
            self.current_board = board_copy
            self.current_color = color_to_move

            # Check for terminal conditions
            if board_copy.is_win(self.color) == self.color:
                return +1.0
            elif board_copy.is_win(self.opponent[self.color]) == self.opponent[self.color]:
                return -1.0

            movesets = board_copy.get_all_possible_moves(color_to_move)
            if not movesets:
                if color_to_move == self.color:
                    return -1.0
                else:
                    return +1.0

            rollout_depth += 1
            if rollout_depth > self.max_rollout_depth:
                return self._heuristic_evaluation(board_copy)

            move = self._select_move(movesets)
            board_copy.make_move(move, color_to_move)
            color_to_move = self.opponent[color_to_move]

    def _select_move(self, movesets):
        """
        Choose a move from the provided movesets with the following priorities:
          1. Prefer moves that do not allow the opponent to have a capturing move.
          2. Among moves, favor those that are capturing moves,
             especially if the move results in promotion to a king.
          3. If no “safe” moves exist, fall back to one of the capturing moves;
             otherwise, pick a random move.
        """
        # Flatten the list-of-lists movesets into one list of candidate moves.
        candidate_moves = []
        for group in movesets:
            for m in group:
                candidate_moves.append(m)

        # Helper to decide if a move is a capturing move.
        def is_capture(move):
            if len(move) < 2:
                return False
            # In many checkers games, a capture move "jumps" over an opponent piece.
            # We assume that a jump will change the row by at least 2.
            return abs(move[1][0] - move[0][0]) >= 2

        # Helper to decide if a move promotes a piece to king.
        def is_promotion(move, color):
            # Assume move is a list of positions (tuples), where the last is the destination.
            dest = move[-1]
            # For color 1, reaching the top row (row index 0) might grant a king.
            if color == 1 and dest[0] == 0:
                return True
            # For color 2, reaching the bottom row (row index self.row-1) might grant a king.
            if color == 2 and dest[0] == (self.row - 1):
                return True
            return False

        safe_moves = []
        capturing_moves = []
        # Evaluate each candidate move.
        for move in candidate_moves:
            # Create a temporary board from the current simulation board.
            temp_board = copy.deepcopy(self.current_board)
            temp_board.make_move(move, self.current_color)

            # Check if this move leaves an opening for the opponent to capture.
            opponent_moves = temp_board.get_all_possible_moves(self.opponent[self.current_color])
            move_allows_capture = False
            for group in opponent_moves:
                for opp_move in group:
                    if is_capture(opp_move):
                        move_allows_capture = True
                        break
                if move_allows_capture:
                    break

            if not move_allows_capture:
                safe_moves.append(move)

            # Also, record capturing moves with a bonus if they result in promotion.
            if is_capture(move):
                bonus = 1
                if is_promotion(move, self.current_color):
                    bonus = 2
                capturing_moves.append((move, bonus))

        # Selection logic: prefer safe moves.
        if safe_moves:
            return random.choice(safe_moves)
        # Otherwise, if we have capturing moves, choose one with the highest bonus.
        elif capturing_moves:
            max_bonus = max(bonus for _, bonus in capturing_moves)
            best_moves = [move for move, bonus in capturing_moves if bonus == max_bonus]
            return random.choice(best_moves)
        else:
            # Fall back to any candidate move if no moves meet our criteria.
            return random.choice(candidate_moves)


    def _heuristic_evaluation(self, board):
        """
        Quick evaluation function if we truncate the rollout.
        Returns +1 if we look better, -1 if we look worse, 0 if roughly balanced.
        """
        # Simple: compare piece counts
        # +1 if we are ahead, -1 if behind, 0 if equal
        my_pieces = 0
        opp_pieces = 0
        for row in board.board:
            for piece in row:
                if piece is not None:
                    if piece.get_color() == self.color:
                        my_pieces += 1
                    else:
                        opp_pieces += 1

        if my_pieces > opp_pieces:
            return +1.0
        elif my_pieces < opp_pieces:
            return -1.0
        else:
            return 0.0