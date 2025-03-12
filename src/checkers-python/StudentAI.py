import copy
import time
import random
import math
from BoardClasses import Move, Board

TIME_LIMIT = 20                         # seconds per move
MAX_ITERATION = 3000                    # Maximum iterations for MCTS
EXPLORATION_PARAM = math.sqrt(2)

class StudentAI:
    def __init__(self, col, row, p):
        self.col = col
        self.row = row
        self.p = p
        self.board = Board(col, row, p)
        self.board.initialize_game()
        self.color = ''                 # Will be set on the first move
        self.opponent = {'W': 'B', 'B': 'W'}
        self.root = None                # The current MCTS tree root
        self.time_limit = TIME_LIMIT 
        self.max_iterations = MAX_ITERATION  
        self.exploration_param = EXPLORATION_PARAM

        # Piece values used during board evaluation
        self.piece_values = {
            'W': 1.0,
            'B': 1.0,
            'WK': 2.0,
            'BK': 2.0
        }

    class TreeNode:
        def __init__(self, board_state, player, move=None, parent_node=None):
            self.game_state = board_state   # Current board configuration
            self.move = move                # Move that led to this node
            self.parent = parent_node
            self.children = []
            self.visits = 0
            self.wins = 0
            self.player = player            # Which player is to move in this node
            self.available_moves = self._init_moves()

        def _init_moves(self):
            # Get and flatten moves; prioritize capture moves
            nested_moves = self.game_state.get_all_possible_moves(self.player)
            flat_moves = [m for group in nested_moves for m in group]
            capture_moves = [m for m in flat_moves if len(m.seq) > 2]
            regular_moves = [m for m in flat_moves if len(m.seq) <= 2]
            return capture_moves + regular_moves

    def get_move(self, opponent_move):
        # On the very first call, set the player's color.
        if not self.color:
            self.color = 'B' if not opponent_move else 'W'

        # Apply the opponent’s move (if any) to our board.
        if opponent_move:
            opp = self.opponent[self.color]
            self.board.make_move(opponent_move, opp)

        # Create a new search tree (root node) or update the existing one.
        if not self.root or not self.root.children:
            # Use a deepcopy of the board so simulations don’t affect the actual game.
            self.root = self.TreeNode(copy.deepcopy(self.board), self.color)
        else:
            self._refresh_tree(opponent_move)

        # Run the MCTS algorithm to choose the best move.
        best_node = self._run_mcts()

        if best_node:
            self.board.make_move(best_node.move, self.color)
            self.root = best_node
            return best_node.move
        else:
            # Fallback: if MCTS did not yield a move, select a random move.
            fallback_moves = self.board.get_all_possible_moves(self.color)
            if fallback_moves:
                chosen = random.choice(random.choice(fallback_moves))
                self.board.make_move(chosen, self.color)
                self.root = self.TreeNode(copy.deepcopy(self.board), self.opponent[self.color])
                return chosen
            return Move([])  # No moves available

    def _refresh_tree(self, opp_move):
        # Update the tree root based on the opponent’s move.
        if opp_move:
            for child in self.root.children:
                if self._boards_equal(self.board, child.game_state):
                    self.root = child
                    self.root.parent = None  # Disconnect the parent link
                    return
        self.root = self.TreeNode(copy.deepcopy(self.board), self.color)

    def _boards_equal(self, board_a, board_b):
        if board_a.black_count != board_b.black_count or board_a.white_count != board_b.white_count:
            return False
        for r in range(board_a.row):
            for c in range(board_a.col):
                piece_a = board_a.board[r][c]
                piece_b = board_b.board[r][c]
                if piece_a.color != piece_b.color or piece_a.is_king != piece_b.is_king:
                    return False
        return True

    def _run_mcts(self):
        start_time = time.time()
        iterations = 0

        # If there is only one possible move, choose it immediately.
        if len(self.root.available_moves) == 1 and not self.root.children:
            single_move = self.root.available_moves[0]
            board_copy = copy.deepcopy(self.root.game_state)
            board_copy.make_move(single_move, self.root.player)
            child = self.TreeNode(board_copy, self.opponent[self.root.player], single_move, self.root)
            self.root.children.append(child)
            return child

        while time.time() - start_time < self.time_limit and iterations < self.max_iterations:
            leaf = self._traverse(self.root)
            new_child = self._expand(leaf)
            if new_child is None:
                continue
            result = self._simulate(new_child)
            self._backpropagate(result, new_child)
            iterations += 1

        return self._select_best_move()

    def _traverse(self, node):
        current = node
        # Descend the tree until a node with untried moves is found.
        while current.children and not current.available_moves:
            current = self._choose_best_child(current)
        return current

    def _choose_best_child(self, node):
        best_val = float('-inf')
        best_children = []
        for child in node.children:
            uct = self._compute_uct(child, self.exploration_param)
            if uct > best_val:
                best_val = uct
                best_children = [child]
            elif uct == best_val:
                best_children.append(child)
        return random.choice(best_children)

    def _compute_uct(self, node, exploration):
        if node.visits == 0:
            return float('inf')
        exploit = node.wins / node.visits
        explore = exploration * math.sqrt(math.log(node.parent.visits) / node.visits)
        # Add a small random factor to help break ties.
        return exploit + explore + random.uniform(0, 0.0001)

    def _expand(self, node):
        if not node.available_moves:
            return None
        # Prioritize capture moves if available.
        capture_options = [m for m in node.available_moves if len(m.seq) > 2]
        if capture_options:
            chosen_move = capture_options[0]
            node.available_moves.remove(chosen_move)
        else:
            chosen_move = node.available_moves.pop(0)
        new_state = copy.deepcopy(node.game_state)
        new_state.make_move(chosen_move, node.player)
        child_node = self.TreeNode(new_state, self.opponent[node.player], chosen_move, node)
        node.children.append(child_node)
        return child_node

    def _simulate(self, node):
        sim_board = copy.deepcopy(node.game_state)
        current_player = node.player
        move_count = 0
        max_moves = 100  # Prevent infinite playouts

        while move_count < max_moves:
            outcome = sim_board.is_win(current_player)
            if outcome != 0:
                # A tie is indicated by outcome == -1.
                if outcome == -1:
                    return 0.5
                # Return win/loss from our AI’s perspective.
                if (outcome == 1 and self.color == 'B') or (outcome == 2 and self.color == 'W'):
                    return 1.0
                else:
                    return 0.0

            possible = sim_board.get_all_possible_moves(current_player)
            if not possible:
                return 0.0 if current_player == self.color else 1.0

            all_moves = [m for group in possible for m in group]
            capture_moves = [m for m in all_moves if len(m.seq) > 2]
            if capture_moves:
                chosen = max(capture_moves, key=lambda m: len(m.seq))
            else:
                chosen = self._choose_simulation_move(sim_board, all_moves, current_player)
            sim_board.make_move(chosen, current_player)
            current_player = self.opponent[current_player]
            move_count += 1

        return self._evaluate_board(sim_board)

    def _choose_simulation_move(self, board, moves, player):
        scored_moves = []
        for m in moves:
            score = 0
            start_row = m.seq[0][0]
            end_row = m.seq[-1][0]
            if player == 'W' and end_row < start_row:
                score += (start_row - end_row) / board.row
                if end_row == 0:
                    score += 0.5
            elif player == 'B' and end_row > start_row:
                score += (end_row - start_row) / board.row
                if end_row == board.row - 1:
                    score += 0.5
            score += self._check_safety(board, m, player)
            score += random.uniform(0, 0.1)
            scored_moves.append((m, score))
        best_move = max(scored_moves, key=lambda tup: tup[1])[0]
        return best_move

    def _check_safety(self, board, move, player):
        end_pos = move.seq[-1]
        opponent = self.opponent[player]
        safety = 0
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dr, dc in directions:
            r1, c1 = end_pos[0] + dr, end_pos[1] + dc
            r2, c2 = end_pos[0] - dr, end_pos[1] - dc
            if board.is_in_board(r1, c1) and board.is_in_board(r2, c2):
                if board.board[r1][c1].color == opponent and board.board[r2][c2].color == '.':
                    safety -= 0.3
        if end_pos[0] == 0 or end_pos[0] == board.row - 1:
            safety += 0.2
        if end_pos[1] == 0 or end_pos[1] == board.col - 1:
            safety += 0.2
        return safety

    def _evaluate_board(self, board):
        white_score = 0
        black_score = 0
        for r in range(board.row):
            for c in range(board.col):
                piece = board.board[r][c]
                if piece.color == 'W':
                    bonus = (board.row - r) / board.row
                    white_score += self.piece_values['WK' if piece.is_king else 'W'] + bonus * 0.2
                elif piece.color == 'B':
                    bonus = r / board.row
                    black_score += self.piece_values['BK' if piece.is_king else 'B'] + bonus * 0.2
        total = white_score + black_score
        if total == 0:
            return 0.5
        return white_score / total if self.color == 'W' else black_score / total

    def _backpropagate(self, result, node):
        current = node
        while current:
            current.visits += 1
            if (current.player == self.color and current.parent) or \
               (current.player == self.opponent[self.color] and not current.parent):
                current.wins += result
            else:
                current.wins += (1 - result)
            current = current.parent

    def _select_best_move(self):
        if not self.root.children:
            return None

        # Check if any child node represents an immediate win.
        win_vals = [1, -1] if self.color == 'B' else [2, -1]
        for child in self.root.children:
            if child.game_state.is_win(self.opponent[self.color]) in win_vals:
                return child

        max_visits = -1
        best_nodes = []
        for child in self.root.children:
            if child.visits > max_visits:
                max_visits = child.visits
                best_nodes = [child]
            elif child.visits == max_visits:
                best_nodes.append(child)
        if len(best_nodes) > 1:
            return max(best_nodes, key=lambda n: n.wins / n.visits)
        return best_nodes[0]
