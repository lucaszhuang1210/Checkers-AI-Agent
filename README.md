# Checkers-AI-Agent

## Running the AI

### Step 1: Navigate to the Tools Directory
```
cd Tools/
```
### Step 2: Run the AI
```
python3 AI_Runner.py 7 7 2 l ../src/checkers-python/main.py ./Sample_AIs/Random_AI/main.py
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
`python3 AI_Runner.py {col} {row} {p} l {AI_1_path} {AI_2_path}`

#### **Network AI Match:**
To play across the network, ensure you're connected through the **school VPN** (must be using school Wi-Fi). Then, navigate to `Tools/AI_Runner.py` and run:
`python3 AI_Runner.py n {AI_path}`
