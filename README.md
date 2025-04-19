# Checkers AI Agent

This project implements an AI agent that plays Checkers using the **Monte Carlo Tree Search (MCTS)** algorithm. The AI utilizes MCTS to simulate potential game outcomes and select optimal moves. The implementation follows the standard MCTS process: selection, expansion, simulation, and backpropagation. A UCB-based selection strategy ensures a balance between exploration and exploitation.

## Authors

- [Lucas Zhuang](https://github.com/lucaszhuang1210)  
- [Jason He](https://github.com/jiach14)

## Core Technical Contributions
For more on how the AI works, see the [Checkers AI Report](https://github.com/lucaszhuang1210/Checkers-AI-Agent/blob/main/Checkers_Final_AI_Report.pdf)

### Monte Carlo Tree Search (MCTS)
Implemented a complete MCTS pipeline from scratch, with the following components:
- **Selection:** Chose nodes using Upper Confidence Bound (UCB) to balance exploration vs. exploitation.
- **Expansion:** Added unexplored child nodes for new legal moves.
- **Simulation:** Performed randomized rollouts to evaluate outcomes.
- **Backpropagation:** Updated node values recursively based on simulated results.

This algorithmic design allowed the agent to make strategic decisions without hardcoded rules, generalizing across a wide range of game states.

### Adaptive Time Management
Developed a dynamic computation scheduler to optimize rollout counts per move:
```python
def update_time_management(self, start_time):
    self.total_time_remaining -= time.time() - start_time
    self.time_divisor -= 0.5 - (1 / self.timed_move_count)
    self.timed_move_count += 1
```
- Early-game moves were given more time for deeper exploration.
- Late-game moves executed faster with more refined search results.
- Prevented timeouts while maintaining high search quality.

Below is a sample of the adaptive allocation strategy, assuming a maximum total runtime of **8 minutes (480 seconds)**:

| Move # | Time Remaining (s) | Time Allocated (s) |
|--------|---------------------|---------------------|
| 1      | 480.00              | 24.5                |
| 2      | 460.41              | 24.5                |
| 3      | 441.62              | 24.3                |
| 4      | 423.47              | 24.0                |
| 5      | 405.80              | 23.8                |

### Performance and Memory Optimization
- Reduced simulation overhead by avoiding unnecessary deep copies.
- Designed a lightweight data structure for move validation and board manipulation.
- Ensured consistent tree updates when handling opponent transitions.

### Heuristic-Guided Simulations
- Integrated a scoring heuristic to discourage risky moves.
- Evaluated draw states and minor advantages during simulations.
- Improved mid-to-late game performance without increasing computational complexity.

---

## Running the AI

### Step 1: Navigate to the Tools Directory
```
cd Tools/
```
### Step 2: Run the AI
```
python3 AI_Runner.py 7 7 2 l ../main.py ./Sample_AIs/Random_AI/main.py
```

## Running Your AI

### Manual Mode
After compiling your AI, use the following command to run it in **manual mode**:
```
python3 main.py {col} {row} {p} m {start_player (0 or 1)}
```

### Play Against Other AIs

The shell supports playing against other **local AI shells** written in different programming languages or against other AIs over the **open network**.

#### **Local AI Match:**
To play against another AI locally, run:
```
python3 AI_Runner.py {col} {row} {p} l {AI_1_path} {AI_2_path}
```

#### **Network AI Match:**
To play across the network, ensure you're connected through the **school VPN** (must be using school Wi-Fi). Then, navigate to `Tools/AI_Runner.py` and run:
```
python3 AI_Runner.py n {AI_path}
```
