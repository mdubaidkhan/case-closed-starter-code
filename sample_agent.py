"""
Case Closed Agent — Hybrid Voronoi (1-ply) + Heuristic
- Opponent-aware territory scoring near contact
- Area flood-fill when far (faster, stabler)
- Smart tie-breaking and safer reversals
"""

import os
from flask import Flask, request, jsonify
from collections import deque

app = Flask(__name__)

# Identity
PARTICIPANT = os.getenv("PARTICIPANT", "UbaidK")
AGENT_NAME = os.getenv("AGENT_NAME", "SmartAgent-HybridVoronoi")

# Judge-populated game state
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

# Directions and helpers
DIRS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
OPPOSITE = {"UP":"DOWN","DOWN":"UP","LEFT":"RIGHT","RIGHT":"LEFT"}

def in_bounds_open(board, x, y):
    return 0 <= y < len(board) and 0 <= x < len(board[0]) and board[y][x] == 0

def next_position(pos, direction):
    x, y = pos
    dx, dy = DIRS[direction]
    return (x + dx, y + dy)

def legal_dirs_from(board, x, y):
    moves = []
    for d, (dx, dy) in DIRS.items():
        nx, ny = x + dx, y + dy
        if in_bounds_open(board, nx, ny):
            moves.append(d)
    return moves

def degree(board, x, y):
    return sum(in_bounds_open(board, x+dx, y+dy) for dx, dy in DIRS.values())

def corridor_width(board, x, y):
    # number of open neighbors around (x,y)
    return degree(board, x, y)

def wall_distance(board, x, y):
    H, W = len(board), len(board[0])
    return min(x, y, W-1-x, H-1-y)

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def path_clear(head, direction, board, steps=3):
    x, y = head
    dx, dy = DIRS[direction]
    for _ in range(steps):
        x, y = x + dx, y + dy
        if not in_bounds_open(board, x, y):
            return False
    return True

# ---- Flood-fills ----

def area_score(x, y, board, limit=40):
    if not in_bounds_open(board, x, y):
        return 0
    q = deque([(x, y)])
    seen = set()
    count = 0
    while q and count < limit:
        cx, cy = q.popleft()
        if (cx, cy) in seen:
            continue
        if not in_bounds_open(board, cx, cy):
            continue
        seen.add((cx, cy))
        count += 1
        q.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])
    return count

def compute_distance_map(board, start_xy, max_expansions=400):
    H, W = len(board), len(board[0])
    INF = 10**9
    dist = [[INF]*W for _ in range(H)]
    x0, y0 = start_xy
    if not in_bounds_open(board, x0, y0):
        return dist
    q = deque([(x0, y0)])
    dist[y0][x0] = 0
    expansions = 0
    while q and expansions < max_expansions:
        x, y = q.popleft()
        base = dist[y][x]
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if in_bounds_open(board, nx, ny) and dist[ny][nx] == INF:
                dist[ny][nx] = base + 1
                q.append((nx, ny))
                expansions += 1
                if expansions >= max_expansions:
                    break
    return dist

def territory_score_given_maps(board, our_dist, opp_dist):
    H, W = len(board), len(board[0])
    INF = 10**9
    our_cells = opp_cells = tie_cells = 0
    for y in range(H):
        row = board[y]
        for x in range(W):
            if row[x] != 0:
                continue
            a, b = our_dist[y][x], opp_dist[y][x]
            if a == INF and b == INF:
                continue
            if a < b:
                our_cells += 1
            elif b < a:
                opp_cells += 1
            else:
                tie_cells += 1
    return our_cells, opp_cells, tie_cells

def score_move_voronoi_min(board, our_head_next, opp_head):
    """Adversarial 1-ply: our next vs all opponent next; take worst outcome for us."""
    opp_dirs = legal_dirs_from(board, opp_head[0], opp_head[1])
    # If opponent is stuck, huge advantage: just count our reachable and add a bonus.
    if not opp_dirs:
        our_dist = compute_distance_map(board, our_head_next, max_expansions=400)
        our_cells = sum(
            1 for y in range(len(board)) for x in range(len(board[0]))
            if board[y][x]==0 and our_dist[y][x] < 10**9
        )
        return our_cells + 1000

    # Precompute our map once for this candidate
    our_dist = compute_distance_map(board, our_head_next, max_expansions=400)

    worst = float('inf')
    for od in opp_dirs:
        ox, oy = next_position(opp_head, od)
        if not in_bounds_open(board, ox, oy):
            continue
        opp_dist = compute_distance_map(board, (ox, oy), max_expansions=400)
        our_cells, opp_cells, tie_cells = territory_score_given_maps(board, our_dist, opp_dist)
        # weights
        alpha = 1.0
        beta = 0.2
        s = our_cells - alpha*opp_cells + beta*tie_cells
        # extra safety near potential head-on encounter
        if manhattan(our_head_next, (ox, oy)) <= 2:
            s -= 10
        worst = min(worst, s)
    return worst

# ---- Decision ----

def decide_move(my_trail, other_trail, turn_count, my_boosts):
    board = game_state.get("board")
    if board is None:
        return "RIGHT"

    head = my_trail[-1] if my_trail else (0, 0)
    opp_head = other_trail[-1] if other_trail else None

    # Current direction (no torus normalization assumed)
    current_dir = "RIGHT"
    if len(my_trail) >= 2:
        prev = my_trail[-2]
        dx = head[0] - prev[0]
        dy = head[1] - prev[1]
        if dx == 1: current_dir = "RIGHT"
        elif dx == -1: current_dir = "LEFT"
        elif dy == 1: current_dir = "DOWN"
        elif dy == -1: current_dir = "UP"

    # Build legal candidates; only drop reverse if there's another option
    all_legal = [d for d in DIRS if in_bounds_open(board, *next_position(head, d))]
    if not all_legal:
        # no legal step; keep heading (may crash, but nothing else)
        best_dir = current_dir
        use_boost = False
        return best_dir

    directions = list(all_legal)
    opp_of_current = OPPOSITE.get(current_dir)
    if opp_of_current in directions and len(directions) > 1:
        directions.remove(opp_of_current)

    # Primary scoring: Voronoi near opponent, area score when far
    CLOSE_THRESH = 6

    best = None
    best_primary = None

    for d in directions:
        nx, ny = next_position(head, d)
        if not in_bounds_open(board, nx, ny):
            continue

        # choose primary score
        if opp_head and manhattan(head, opp_head) <= CLOSE_THRESH:
            primary = score_move_voronoi_min(board, (nx, ny), opp_head)
        else:
            primary = area_score(nx, ny, board, limit=40)

        # tie-break features (higher is better):
        straight = 1 if d == current_dir else 0
        deg = degree(board, nx, ny)
        # corridor lookahead 1 step forward in the same direction
        fx, fy = nx + DIRS[d][0], ny + DIRS[d][1]
        w1 = corridor_width(board, nx, ny)
        w2 = corridor_width(board, fx, fy) if in_bounds_open(board, fx, fy) else 0
        wd = wall_distance(board, nx, ny)
        # tiny deterministic jitter to break perfect symmetry
        jitter = ((nx * 73856093) ^ (ny * 19349663) ^ (turn_count * 83492791)) & 7
        jitter *= 0.01

        cand = (primary, straight, deg, w1 + w2, wd, jitter, d)

        if best is None or cand > best:
            best = cand
            best_primary = primary

    if best is None:
        # fall back to any legal (including reverse if needed)
        best_dir = all_legal[0]
        use_boost = False
        return best_dir

    best_dir = best[-1]

    # Boost only when it's smart and safe
    use_boost = (
        my_boosts > 0
        and 30 <= turn_count <= 80
        and path_clear(head, best_dir, board, steps=3)
        and (best_primary is not None and best_primary > 5)
    )

    return f"{best_dir}:BOOST" if use_boost else best_dir

# ---- Flask endpoints ----

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5009"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
