import os
import uuid
import time
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(_name_)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX_Minimax" # Renamed agent

# --- AI Agent Constants ---
HEIGHT = 18
WIDTH = 20
EMPTY = 0
AGENT_CELL = 1 # Assuming 1 represents any agent/trail

# Max time in milliseconds to think. Set to ~150-200ms for competition
# For local testing with judge_engine.py (4s timeout), you can set this higher
THINK_TIME_MS = 150 

# Move mappings
DIRECTIONS_MAP = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}
OPPOSITE_MOVE = {
    "UP": "DOWN",
    "DOWN": "UP",
    "LEFT": "RIGHT",
    "RIGHT": "LEFT"
}
# --------------------------


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    This is essential for tracking our agent's last move (direction).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]["grid"]
                GLOBAL_GAME.agent1.trail = deque(
                    [tuple(pos) for pos in data.get("agent1_trail", [])]
                )
                GLOBAL_GAME.agent2.trail = deque(
                    [tuple(pos) for pos in data.get("agent2_trail", [])]
                )
                GLOBAL_GAME.agent1.length = data.get("agent1_length", 1)
                GLOBAL_GAME.agent2.length = data.get("agent2_length", 1)
                GLOBAL_GAME.agent1.alive = data.get("agent1_alive", True)
                GLOBAL_GAME.agent2.alive = data.get("agent2_alive", True)
                GLOBAL_GAME.agent1.boosts_remaining = data.get("agent1_boosts", 3)
                GLOBAL_GAME.agent2.boosts_remaining = data.get("agent2_boosts", 3)
                GLOBAL_GAME.turns = data.get("turn_count", 0)

                # CRITICAL: Update agent directions based on last move
                # The state payload only gives opponent's last direction
                # We must infer our own from the game state we track
                if "agent1_last_direction" in data:
                    dir_str = data["agent1_last_direction"]
                    GLOBAL_GAME.agent1.direction = Direction[dir_str] if dir_str else None
                if "agent2_last_direction" in data:
                    dir_str = data["agent2_last_direction"]
                    GLOBAL_GAME.agent2.direction = Direction[dir_str] if dir_str else None

            except Exception as e:
                print(f"Error updating local game: {e}")
                # Fallback: re-initialize
                GLOBAL_GAME.reset()


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400
    
    _update_local_game_from_post(data)
    
    return jsonify({"status": "ok"}), 200


# --- AI Logic Functions ---

def get_valid_moves(grid: list[list[int]], pos: tuple[int, int], last_move_dir_str: str | None) -> dict[str, tuple[int, int]]:
    """
    Gets all valid (non-suicidal) moves from a given position.
    Returns a dictionary of {"MOVE_STR": (new_y, new_x)}
    """
    valid_moves = {}
    opposite_move = OPPOSITE_MOVE.get(last_move_dir_str)
    
    for move_str, (dy, dx) in DIRECTIONS_MAP.items():
        if move_str == opposite_move:
            continue
            
        new_y = (pos[0] + dy) % HEIGHT
        new_x = (pos[1] + dx) % WIDTH
        
        if grid[new_y][new_x] == EMPTY:
            valid_moves[move_str] = (new_y, new_x)
            
    return valid_moves

def _flood_fill(grid: list[list[int]], start_pos: tuple[int, int]) -> int:
    """
    Calculates the number of reachable empty cells from a start position.
    """
    q = deque([start_pos])
    visited = set([start_pos])
    count = 0
    
    while q:
        y, x = q.popleft()
        count += 1
        
        for (dy, dx) in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ny = (y + dy) % HEIGHT
            nx = (x + dx) % WIDTH
            
            if (ny, nx) not in visited and grid[ny][nx] == EMPTY:
                visited.add((ny, nx))
                q.append((ny, nx))
                
    return count

def heuristic_flood_fill(grid: list[list[int]], my_pos: tuple[int, int], opp_pos: tuple[int, int]) -> int:
    """
    Heuristic function. Calculates "territory" difference.
    A larger, positive score is better for us.
    """
    # Create a temporary copy of the grid to avoid modifying the one in recursion
    # This is safer for the flood fill, which assumes our positions are "empty"
    temp_grid = [row[:] for row in grid]
    if 0 <= my_pos[0] < HEIGHT and 0 <= my_pos[1] < WIDTH:
        temp_grid[my_pos[0]][my_pos[1]] = EMPTY
    if 0 <= opp_pos[0] < HEIGHT and 0 <= opp_pos[1] < WIDTH:
        temp_grid[opp_pos[0]][opp_pos[1]] = EMPTY

    my_space = _flood_fill(temp_grid, my_pos)
    opp_space = _flood_fill(temp_grid, opp_pos)
    
    # Simple "Voronoi" or territory-based heuristic
    return my_space - opp_space

def minimax(
    grid: list[list[int]], 
    my_pos: tuple[int, int], 
    opp_pos: tuple[int, int], 
    my_last_dir_str: str | None, 
    opp_last_dir_str: str | None, 
    is_maximizing_player: bool, 
    depth: int, 
    alpha: float, 
    beta: float,
    tt: dict
) -> tuple[str | None, int]:
    """
    Minimax search algorithm with alpha-beta pruning.
    """
    
    # Check transposition table (cache)
    state_key = (my_pos, opp_pos, my_last_dir_str, opp_last_dir_str, is_maximizing_player, depth)
    if state_key in tt:
        return tt[state_key]

    # Get valid moves for both players to check for terminal states
    my_valid_moves = get_valid_moves(grid, my_pos, my_last_dir_str)
    opp_valid_moves = get_valid_moves(grid, opp_pos, opp_last_dir_str)

    # --- Terminal State Checks ---
    if not my_valid_moves:
        # We have no moves, we lose (unless opponent also has no moves)
        return (None, -1000000 + depth) # Add depth to prefer losing later
    if not opp_valid_moves:
        # Opponent has no moves, we win
        return (None, 1000000 - depth) # Subtract depth to prefer winning sooner
    
    if depth == 0:
        # Reached max depth, evaluate board
        score = heuristic_flood_fill(grid, my_pos, opp_pos)
        return (None, score)

    # --- Recursive Search ---
    
    if is_maximizing_player:
        best_score = -float('inf')
        best_move_str = list(my_valid_moves.keys())[0] # Default to first move

        # Move ordering: simple, no ordering for now
        for move_str, new_pos in my_valid_moves.items():
            
            grid[new_pos[0]][new_pos[1]] = AGENT_CELL # Simulate move
            
            _score = minimax(
                grid, new_pos, opp_pos, move_str, opp_last_dir_str, 
                False, depth - 1, alpha, beta, tt
            )[1]
            
            grid[new_pos[0]][new_pos[1]] = EMPTY # Backtrack
            
            if _score > best_score:
                best_score = _score
                best_move_str = move_str
                
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break # Pruning
        
        tt[state_key] = (best_move_str, best_score)
        return (best_move_str, best_score)

    else: # Minimizing player (Opponent's turn)
        best_score = float('inf')
        best_move_str = list(opp_valid_moves.keys())[0] # Default

        for move_str, new_pos in opp_valid_moves.items():
            
            grid[new_pos[0]][new_pos[1]] = AGENT_CELL # Simulate move
            
            _score = minimax(
                grid, my_pos, new_pos, my_last_dir_str, move_str, 
                True, depth - 1, alpha, beta, tt
            )[1]
            
            grid[new_pos[0]][new_pos[1]] = EMPTY # Backtrack
            
            if _score < best_score:
                best_score = _score
                best_move_str = move_str
            
            beta = min(beta, best_score)
            if beta <= alpha:
                break # Pruning
        
        tt[state_key] = (best_move_str, best_score)
        return (best_move_str, best_score)

def iterative_deepening(
    grid: list[list[int]], 
    my_pos: tuple[int, int], 
    opp_pos: tuple[int, int], 
    my_last_dir_str: str | None, 
    opp_last_dir_str: str | None, 
    start_time: float
) -> str:
    """
    Controller function for Iterative Deepening Depth-First Search (IDDFS).
    Calls minimax with increasing depth until time runs out.
    """
    
    # Get moves available from our current position
    initial_valid_moves = get_valid_moves(grid, my_pos, my_last_dir_str)
    
    if not initial_valid_moves:
        print("AGENT: No valid moves! Defaulting to UP.")
        return "UP" # We are trapped, return anything
        
    # Default to the first valid move
    best_move = list(initial_valid_moves.keys())[0]
    
    depth = 1
    
    while True:
        time_elapsed = (time.time() - start_time) * 1000
        if time_elapsed > THINK_TIME_MS:
            print(f"AGENT: Time out. Best move from depth {depth - 1}: {best_move}")
            break # Time's up, return the best move from the last completed depth
        
        print(f"AGENT: Starting search for depth {depth}...")
        
        # Transposition table (cache) for this depth search
        transposition_table = {}
        
        # --- Top-level minimax loop (we are maximizing player) ---
        current_best_move_for_depth = best_move
        current_best_score = -float('inf')
        
        # --- Move Ordering ---
        # Sort moves: try the best move from the previous depth first
        ordered_move_strs = list(initial_valid_moves.keys())
        if best_move in ordered_move_strs:
            ordered_move_strs.remove(best_move)
            ordered_move_strs.insert(0, best_move)
            
        for move_str in ordered_move_strs:
            # Check for timeout before starting the next recursive call
            if (time.time() - start_time) * 1000 > THINK_TIME_MS:
                print("AGENT: Timeout during move evaluation, breaking loop.")
                break # Go to outer loop, which will also break
                
            new_pos = initial_valid_moves[move_str]
            
            grid[new_pos[0]][new_pos[1]] = AGENT_CELL # Simulate move
            
            # Call minimax for the opponent's turn (is_maximizing=False)
            _score = minimax(
                grid, new_pos, opp_pos, move_str, opp_last_dir_str,
                False, depth - 1, -float('inf'), float('inf'), transposition_table
            )[1]
            
            grid[new_pos[0]][new_pos[1]] = EMPTY # Backtrack
            
            if _score > current_best_score:
                current_best_score = _score
                current_best_move_for_depth = move_str
        
        # Check time again after completing the depth
        if (time.time() - start_time) * 1000 > THINK_TIME_MS:
             print(f"AGENT: Finished depth {depth} but timed out. Using previous best.")
             # We finished the loop but went over time, so the best_move
             # from the previous depth is the last guaranteed-in-time move.
             break
        
        # This depth completed in time. Save its best move.
        best_move = current_best_move_for_depth
        print(f"AGENT: Finished depth {depth}. Best move: {best_move} (Score: {current_best_score})")
        
        depth += 1
        
    return best_move

# --- Main Move Endpoint ---

@app.route("/move", methods=["POST"])
def move():
    """
    Judge calls this to request a move from the agent.
    Agent must respond with {"move": "DIRECTION"}
    """
    start_time = time.time()
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)
        # Get agent objects from our local game copy
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        opp_agent = GLOBAL_GAME.agent2 if player_number == 1 else GLOBAL_GAME.agent1
        
        # Get last move direction as a string ("UP", "DOWN", etc.)
        my_last_move_dir = my_agent.direction.name if my_agent.direction else None
        opp_last_move_dir = opp_agent.direction.name if opp_agent.direction else None
        
        # Not using boosts for this agent, but good to have
        boosts_remaining = my_agent.boosts_remaining
   
    # ----------------- AI code starts here -------------------
    
    # 1. Parse the board state
    grid = state.get("board", {}).get("grid", [])
    if not grid:
        print("AGENT: No grid in state! Defaulting to UP.")
        return jsonify({"move": "UP"}), 200 # Failsafe
        
    # 2. Get player positions
    my_trail = state.get(f"agent{player_number}_trail", [])
    if not my_trail:
        print("AGENT: No my_trail in state! Defaulting to UP.")
        return jsonify({"move": "UP"}), 200 # Failsafe
    my_pos = tuple(my_trail[0])
    
    opp_player_num = 2 if player_number == 1 else 1
    opp_trail = state.get(f"agent{opp_player_num}_trail", [])
    if not opp_trail:
        print("AGENT: No opp_trail in state! Defaulting to UP.")
        return jsonify({"move": "UP"}), 200 # Failsafe
    opp_pos = tuple(opp_trail[0])
    
    # 3. Handle edge case: opponent is on our head (simultaneous crash)
    # The search algorithm will see this as a valid move, but it's a loss/draw
    # It's better to let the search algorithm handle this,
    # as it will be pruned if it's a bad move.

    # 4. Run the AI
    # Create a copy of the grid for the AI to simulate on
    grid_copy = [row[:] for row in grid]
    
    best_move = iterative_deepening(
        grid_copy, 
        my_pos, 
        opp_pos, 
        my_last_move_dir, 
        opp_last_dir_str, 
        start_time
    )
    
    # 5. Final Move
    # Note: This agent does not use boosts. To implement boosts,
    # you would need to modify the get_valid_moves and minimax
    # functions to account for a 2-step move.
    move = best_move
    
    # -----------------end code here--------------------
    
    time_taken = (time.time() - start_time) * 1000
    print(f"AGENT: Final move: {move}. Time taken: {time_taken:.2f} ms")
    
    # Update our local game state with our intended move
    # The judge will confirm this in the next /send-state
    with game_lock:
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        if move in Direction._members_:
            my_agent.direction = Direction[move]

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state."""
    data = request.get_json()
    _update_local_game_from_post(data)
    
    result = data.get("result", "UNKNOWN")
    print(f"AGENT: Game over. Result: {result}")
    
    # Reset local game for the next match
    GLOBAL_GAME.reset()
    
    return jsonify({"status": "ok"}), 200


if _name_ == "_main_":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)