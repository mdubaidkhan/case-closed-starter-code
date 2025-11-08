"""
Improved Agent for Case Closed Challenge
Adds flood-fill space awareness, safer boosts, and opponent avoidance.
"""

import os
from flask import Flask, request, jsonify
from collections import deque

app = Flask(__name__)

# Basic identity
PARTICIPANT = os.getenv("PARTICIPANT", "UbaidK")
AGENT_NAME = os.getenv("AGENT_NAME", "SmartAgent")

# Track game state
game_state = {
    "board": None,
    "agent1_trail": [],
    "agent2_trail": [],
    "agent1_length": 0,
    "agent2_length": 0,
    "agent1_alive": True,
    "agent2_alive": True,
    "agent1_boosts": 3,
    "agent2_boosts": 3,
    "turn_count": 0,
    "player_number": 1,
}


@app.route("/", methods=["GET"])
def info():
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


@app.route("/send-state", methods=["POST"])
def receive_state():
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    game_state.update(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    player_number = request.args.get("player_number", default=1, type=int)
    turn_count = game_state.get("turn_count", 0)

    if player_number == 1:
        my_trail = game_state.get("agent1_trail", [])
        my_boosts = game_state.get("agent1_boosts", 3)
        other_trail = game_state.get("agent2_trail", [])
    else:
        my_trail = game_state.get("agent2_trail", [])
        my_boosts = game_state.get("agent2_boosts", 3)
        other_trail = game_state.get("agent1_trail", [])

    move = decide_move(my_trail, other_trail, turn_count, my_boosts)
    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    data = request.get_json()
    if data:
        result = data.get("result", "UNKNOWN")
        print(f"\nGame Over! Result: {result}")
    return jsonify({"status": "acknowledged"}), 200


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def next_position(pos, direction):
    """Compute next (x, y) given a direction."""
    x, y = pos
    if direction == "UP":
        return (x, y - 1)
    elif direction == "DOWN":
        return (x, y + 1)
    elif direction == "LEFT":
        return (x - 1, y)
    elif direction == "RIGHT":
        return (x + 1, y)
    return pos


def area_score(x, y, board, limit=40):
    """BFS flood-fill to estimate open space reachable from (x, y)."""
    if not (0 <= y < len(board) and 0 <= x < len(board[0])):
        return 0
    if board[y][x] != 0:
        return 0

    q = deque([(x, y)])
    visited = set()
    count = 0

    while q and count < limit:
        cx, cy = q.popleft()
        if (cx, cy) in visited:
            continue
        if not (0 <= cy < len(board) and 0 <= cx < len(board[0])):
            continue
        if board[cy][cx] != 0:
            continue

        visited.add((cx, cy))
        count += 1

        # Add neighbors
        q.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])
    return count


def path_clear(head, direction, board, steps=3):
    """Check if path ahead is clear for a few cells."""
    x, y = head
    for _ in range(steps):
        x, y = next_position((x, y), direction)
        if not (0 <= y < len(board) and 0 <= x < len(board[0])):
            return False
        if board[y][x] != 0:
            return False
    return True


# ------------------------------------------------------------
# Main decision logic
# ------------------------------------------------------------

def decide_move(my_trail, other_trail, turn_count, my_boosts):
    """Improved move decision with open-space scoring and safety."""
    board = game_state.get("board")
    if board is None:
        return "RIGHT"

    head = my_trail[-1] if my_trail else (0, 0)

    # Determine current direction
    current_dir = "RIGHT"
    if len(my_trail) >= 2:
        prev = my_trail[-2]
        dx = head[0] - prev[0]
        dy = head[1] - prev[1]
        if dx == 1: current_dir = "RIGHT"
        elif dx == -1: current_dir = "LEFT"
        elif dy == 1: current_dir = "DOWN"
        elif dy == -1: current_dir = "UP"

    # Get all directions except direct reverse
    directions = ["UP", "DOWN", "LEFT", "RIGHT"]
    opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    if current_dir in opposite and opposite[current_dir] in directions:
        directions.remove(opposite[current_dir])

    # Evaluate moves
    best_dir = current_dir
    best_score = -1
    opp_head = other_trail[-1] if other_trail else None

    for d in directions:
        nx, ny = next_position(head, d)

        # Skip if out of bounds or blocked
        if not (0 <= ny < len(board) and 0 <= nx < len(board[0])):
            continue
        if board[ny][nx] != 0:
            continue

        # Compute flood-fill area
        score = area_score(nx, ny, board)

        # Penalize proximity to opponent head
        if opp_head:
            dist = abs(nx - opp_head[0]) + abs(ny - opp_head[1])
            if dist <= 2:
                score -= 10  # avoid danger zones

        # Prefer continuing straight slightly
        if d == current_dir:
            score += 2

        if score > best_score:
            best_score = score
            best_dir = d

    # Safe boost usage
    use_boost = (
        my_boosts > 0
        and 30 <= turn_count <= 80
        and path_clear(head, best_dir, board)
        and best_score > 5
    )

    return f"{best_dir}:BOOST" if use_boost else best_dir


# ------------------------------------------------------------
# Run server
# ------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
