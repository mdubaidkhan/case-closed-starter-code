"""
Case Closed Agent — HYBRID: Expand → Fight → Fill
- Early expansion with boost + center bias
- Contested: adversarial 1-ply Voronoi + cut/separation bonus
- Solo: Hamiltonian cycle follow (safe fill)
- Safe reverse rule, robust tie-breakers, safe boost gate (engine boost=2; check 3 ahead)

Flask server compatible with the Judge.
"""

import os
from flask import Flask, request, jsonify
from collections import deque

app = Flask(__name__)

# Identity
PARTICIPANT = os.getenv("PARTICIPANT", "UbaidK")
AGENT_NAME  = os.getenv("AGENT_NAME",  "SmartAgent-HYBRID")

# Judge-populated state
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

# Directions
DIRS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
OPPOSITE = {"UP":"DOWN","DOWN":"UP","LEFT":"RIGHT","RIGHT":"LEFT"}
INF = 10**9

# ----------------------------
# Basic helpers
# ----------------------------

def in_bounds(board, x, y):
    return 0 <= y < len(board) and 0 <= x < len(board[0])

def open_cell(board, x, y):
    return in_bounds(board, x, y) and board[y][x] == 0

def next_pos_xy(x, y, d):
    dx, dy = DIRS[d]
    return x + dx, y + dy

def degree(board, x, y):
    return sum(open_cell(board, x+dx, y+dy) for dx,dy in DIRS.values())

def wall_distance(board, x, y):
    H, W = len(board), len(board[0])
    return min(x, y, W-1-x, H-1-y)

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def path_clear(head, d, board, steps=3):
    x, y = head
    dx, dy = DIRS[d]
    for _ in range(steps):
        x += dx; y += dy
        if not open_cell(board, x, y):
            return False
    return True

# ----------------------------
# BFS / distances / components
# ----------------------------

def area_score(x, y, board, limit=60):
    if not open_cell(board, x, y): return 0
    q = deque([(x, y)])
    seen = set()
    cnt = 0
    while q and cnt < limit:
        cx, cy = q.popleft()
        if (cx, cy) in seen: continue
        if not open_cell(board, cx, cy): continue
        seen.add((cx, cy))
        cnt += 1
        for dx,dy in DIRS.values():
            q.append((cx+dx, cy+dy))
    return cnt

def flood_component_size(board, sx, sy, limit=None):
    if not open_cell(board, sx, sy): return 0
    q = deque([(sx, sy)])
    seen = {(sx, sy)}
    cnt = 0
    while q:
        x,y = q.popleft()
        cnt += 1
        if limit and cnt >= limit: return cnt
        for dx,dy in DIRS.values():
            nx,ny = x+dx, y+dy
            if open_cell(board, nx, ny) and (nx,ny) not in seen:
                seen.add((nx,ny)); q.append((nx,ny))
    return cnt

def compute_distance_map(board, start_xy, max_expansions=400):
    H, W = len(board), len(board[0])
    dist = [[INF]*W for _ in range(H)]
    x0,y0 = start_xy
    if not open_cell(board, x0, y0): return dist
    q = deque([(x0,y0)])
    dist[y0][x0] = 0
    exp = 0
    while q and exp < max_expansions:
        x,y = q.popleft()
        base = dist[y][x]
        for dx,dy in DIRS.values():
            nx,ny = x+dx, y+dy
            if open_cell(board, nx, ny) and dist[ny][nx] == INF:
                dist[ny][nx] = base + 1
                q.append((nx,ny))
                exp += 1
                if exp >= max_expansions: break
    return dist

def territory_score_given_maps(board, our_dist, opp_dist):
    H,W = len(board), len(board[0])
    our_cells = opp_cells = ties = 0
    for y in range(H):
        row = board[y]
        for x in range(W):
            if row[x] != 0: continue
            a,b = our_dist[y][x], opp_dist[y][x]
            if a == INF and b == INF: continue
            if a < b: our_cells += 1
            elif b < a: opp_cells += 1
            else: ties += 1
    return our_cells, opp_cells, ties

def heads_connected(board, our_head, opp_head):
    """Is there a free path between heads?"""
    if not (open_cell(board, *our_head) and open_cell(board, *opp_head)):
        return False
    q = deque([our_head])
    seen = {our_head}
    while q:
        x,y = q.popleft()
        if (x,y) == opp_head: return True
        for dx,dy in DIRS.values():
            nx,ny = x+dx, y+dy
            if open_cell(board, nx, ny) and (nx,ny) not in seen:
                seen.add((nx,ny)); q.append((nx,ny))
    return False

# ----------------------------
# Hamiltonian cycle (serpentine) for even HxW
# ----------------------------

def build_serpentine_cycle(W, H):
    order = []
    for y in range(H):
        row = list(range(W))
        if y % 2 == 1:
            row.reverse()
        for x in row:
            order.append((x, y))
    nxt = {}
    for i,(x,y) in enumerate(order):
        nx,ny = order[(i+1) % len(order)]
        nxt[(x,y)] = (nx,ny)
    return nxt

_CYCLE_CACHE = {}
def get_cycle_map(board):
    H, W = len(board), len(board[0])
    key = (W,H)
    if key not in _CYCLE_CACHE:
        _CYCLE_CACHE[key] = build_serpentine_cycle(W,H)
    return _CYCLE_CACHE[key]

def cycle_next_dir(nxt_map, x, y):
    nx, ny = nxt_map[(x,y)]
    dx, dy = nx - x, ny - y
    for d,(ox,oy) in DIRS.items():
        if (ox,oy) == (dx,dy):
            return d
    # Shouldn't happen on torus serpentine; fallback:
    return "RIGHT"

# ----------------------------
# Scoring (contested)
# ----------------------------

def score_move_voronoi_min(board, our_head_next, opp_head):
    """Adversarial 1-ply territory with head-on caution."""
    opp_dirs_xy = []
    for d,(dx,dy) in DIRS.items():
        ox, oy = opp_head[0]+dx, opp_head[1]+dy
        if open_cell(board, ox, oy): opp_dirs_xy.append((ox,oy))
    if not opp_dirs_xy:
        our_dist = compute_distance_map(board, our_head_next, max_expansions=400)
        our_cells = sum(1 for y in range(len(board)) for x in range(len(board[0]))
                        if board[y][x]==0 and our_dist[y][x] < INF)
        return our_cells + 1000

    our_dist = compute_distance_map(board, our_head_next, max_expansions=400)
    worst = float('inf')
    for (ox,oy) in opp_dirs_xy:
        opp_dist = compute_distance_map(board, (ox,oy), max_expansions=400)
        oc, pc, tc = territory_score_given_maps(board, our_dist, opp_dist)
        s = oc - 1.0*pc + 0.2*tc
        if manhattan(our_head_next, (ox,oy)) <= 2: s -= 10
        worst = min(worst, s)
    return worst

def cut_bonus(board, opp_head, nx, ny):
    """Pretend we occupy (nx,ny) -> how much does opponent component shrink?"""
    if not open_cell(board, nx, ny): return 0.0
    base = flood_component_size(board, opp_head[0], opp_head[1])
    if base == 0: return 0.0
    saved = board[ny][nx]
    board[ny][nx] = 1
    after = flood_component_size(board, opp_head[0], opp_head[1])
    board[ny][nx] = saved
    gain = base - after
    near_deg = degree(board, opp_head[0], opp_head[1])
    choke = 1.0 if near_deg <= 2 else 0.0
    return (3.0 * gain + 5.0 * choke) if gain > 0 else (1.5 * choke)

# ----------------------------
# Decision
# ----------------------------

def decide_move(my_trail, other_trail, turn_count, my_boosts):
    board = game_state.get("board")
    if board is None: return "RIGHT"

    head = my_trail[-1] if my_trail else (0,0)
    opp_head = other_trail[-1] if other_trail else None

    # current dir
    current_dir = "RIGHT"
    if len(my_trail) >= 2:
        prev = my_trail[-2]
        dx, dy = head[0]-prev[0], head[1]-prev[1]
        if   dx == 1: current_dir = "RIGHT"
        elif dx == -1: current_dir = "LEFT"
        elif dy == 1: current_dir = "DOWN"
        elif dy == -1: current_dir = "UP"

    # legal candidates (keep reverse if it's the only option)
    all_legal = [d for d in DIRS if open_cell(board, *next_pos_xy(*head, d))]
    if not all_legal:
        return current_dir
    directions = list(all_legal)
    opp_of_current = OPPOSITE.get(current_dir)
    if opp_of_current in directions and len(directions) > 1:
        directions.remove(opp_of_current)

    # Phase detection
    EARLY = turn_count < 30
    close = (opp_head is not None and manhattan(head, opp_head) <= 6)
    connected = (opp_head is not None and heads_connected(board, head, opp_head))
    SOLO = (opp_head is not None and not connected)

    # SOLO MODE: follow Hamiltonian cycle
    if SOLO:
        nxt_map = get_cycle_map(board)
        d_cycle = cycle_next_dir(nxt_map, head[0], head[1])
        nx, ny = next_pos_xy(*head, d_cycle)
        if open_cell(board, nx, ny):
            use_boost = (my_boosts > 0 and 30 <= turn_count <= 80
                         and path_clear(head, d_cycle, board, steps=3))
            return f"{d_cycle}:BOOST" if use_boost else d_cycle
        # If blocked, fall through to normal scoring

    # Scoring per phase
    best = None
    best_primary = None

    for d in directions:
        nx, ny = next_pos_xy(*head, d)
        if not open_cell(board, nx, ny): continue

        if opp_head and connected and close:
            # CONTESTED: Voronoi + cut bonus
            primary = score_move_voronoi_min(board, (nx, ny), opp_head)
            primary += cut_bonus(board, opp_head, nx, ny)
        elif EARLY and not close:
            # EARLY EXPANSION: larger area cap + center bias
            primary = area_score(nx, ny, board, limit=70)
            if wall_distance(board, nx, ny) >= 2:
                primary += 1.0
        else:
            # DEFAULT: moderate area lookahead
            primary = area_score(nx, ny, board, limit=50)
            if not connected and wall_distance(board, nx, ny) >= 2:
                primary += 0.5

        # Tie-breakers (lexicographic)
        straight = 1 if d == current_dir else 0
        deg_now = degree(board, nx, ny)
        fx, fy = next_pos_xy(nx, ny, d)
        w2 = degree(board, fx, fy) if open_cell(board, fx, fy) else 0
        wd = wall_distance(board, nx, ny)
        jitter = ((nx * 73856093) ^ (ny * 19349663) ^ (turn_count * 83492791)) & 7
        jitter *= 0.01

        cand = (primary, straight, deg_now, deg_now + w2, wd, jitter, d)
        if best is None or cand > best:
            best = cand
            best_primary = primary

    if best is None:
        return all_legal[0]

    best_dir = best[-1]

    # Boost policy (safe & phase-aware)
    allow_boost = (
        my_boosts > 0
        and path_clear(head, best_dir, board, steps=3)  # engine boost=2; check 3 for margin
        and (best_primary is not None and best_primary > 5)
    )

    # More aggressive boost early if lane is open & not hugging wall
    if EARLY and not close:
        nx, ny = next_pos_xy(*head, best_dir)
        allow_boost = allow_boost and (wall_distance(board, nx, ny) > 1)

    # Mid-game boost window
    if 30 <= turn_count <= 80 and allow_boost:
        return f"{best_dir}:BOOST"
    else:
        return best_dir

# ----------------------------
# Flask endpoints
# ----------------------------

@app.route("/", methods=["GET"])
def info():
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200

@app.route("/send-state", methods=["POST"])
def receive_state():
    data = request.get_json()
    if not data: return jsonify({"error": "no json body"}), 400
    game_state.update(data)
    return jsonify({"status": "state received"}), 200

@app.route("/send-move", methods=["GET"])
def send_move():
    player_number = request.args.get("player_number", default=1, type=int)
    turn_count = game_state.get("turn_count", 0)

    if player_number == 1:
        my_trail   = game_state.get("agent1_trail", [])
        my_boosts  = game_state.get("agent1_boosts", 3)
        other_trail= game_state.get("agent2_trail", [])
    else:
        my_trail   = game_state.get("agent2_trail", [])
        my_boosts  = game_state.get("agent2_boosts", 3)
        other_trail= game_state.get("agent1_trail", [])

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
