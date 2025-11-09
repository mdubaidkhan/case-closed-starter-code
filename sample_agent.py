"""
Case Closed Agent — HYBRID + SEAL + Local Guard + Axis Alternation + Midline Opening + Anchored Solo
"""

import os
from flask import Flask, request, jsonify
from collections import deque

app = Flask(__name__)

# Identity
PARTICIPANT = os.getenv("PARTICIPANT", "UbaidK")
AGENT_NAME  = os.getenv("AGENT_NAME",  "SmartAgent-HYBRID-AXIS-GUARD-MIDLINE")

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
ORDERED_DIRS = ["UP", "RIGHT", "DOWN", "LEFT"]
OPPOSITE = {"UP":"DOWN","DOWN":"UP","LEFT":"RIGHT","RIGHT":"LEFT"}
LEFT_OF  = {"UP":"LEFT","LEFT":"DOWN","DOWN":"RIGHT","RIGHT":"UP"}
RIGHT_OF = {"UP":"RIGHT","RIGHT":"DOWN","DOWN":"LEFT","LEFT":"UP"}
INF = 10**9

# --- Seal Mode state ---
SEAL_MODE = False
SEAL_PLAN = deque()
SEAL_TARGET = None
SEAL_BLOCKS = 0
SEAL_REPLAN_ONCE = False

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

def center_bias_delta(board, x, y, nx, ny):
    H, W = len(board), len(board[0])
    cx, cy = (W-1)/2.0, (H-1)/2.0
    d0 = abs(x - cx) + abs(y - cy)
    d1 = abs(nx - cx) + abs(ny - cy)
    return 1 if d1 < d0 else 0

def tunnel_len(board, x, y, d, maxn=6):
    dx,dy = DIRS[d]; t=0
    for _ in range(maxn):
        x+=dx; y+=dy
        if not open_cell(board,x,y): break
        if degree(board,x,y) != 2: break
        t+=1
    return t

def rel_order(current_dir, d):
    # forward > left > right > back
    if d == current_dir: return 0
    if d == LEFT_OF[current_dir]: return 1
    if d == RIGHT_OF[current_dir]: return 2
    return 3

def recent_axis_bias(my_trail, k=4):
    if len(my_trail) < k+1:
        return None
    moves = []
    for i in range(-k, 0):
        x2,y2 = my_trail[i]
        x1,y1 = my_trail[i-1]
        dx,dy = x2-x1, y2-y1
        if dx != 0: moves.append("H")
        elif dy != 0: moves.append("V")
    if len(moves) == k and all(m == "H" for m in moves): return "H"
    if len(moves) == k and all(m == "V" for m in moves): return "V"
    return None

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
        seen.add((cx, cy)); cnt += 1
        for dx,dy in DIRS.values():
            q.append((cx+dx, cy+dy))
    return cnt

def flood_component_size_xy(board, sx, sy, limit=None):
    if not open_cell(board, sx, sy): return 0
    q = deque([(sx, sy)])
    seen = {(sx, sy)}
    cnt = 0
    while q:
        x,y = q.popleft(); cnt += 1
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

def shortest_path_dirs(board, start, target):
    if not (open_cell(board, *start) and open_cell(board, *target)):
        return []
    q = deque([start])
    prev = {start: None}
    prev_dir = {}
    while q:
        x,y = q.popleft()
        if (x,y) == target:
            path = []
            cur = target
            while prev[cur] is not None:
                path.append(prev_dir[cur]); cur = prev[cur]
            path.reverse(); return path
        for d,(dx,dy) in DIRS.items():
            nx,ny = x+dx, y+dy
            if open_cell(board, nx, ny) and (nx,ny) not in prev:
                prev[(nx,ny)] = (x,y)
                prev_dir[(nx,ny)] = d
                q.append((nx,ny))
    return []

# ----------------------------
# Serpentine cycles (anchored by half)
# ----------------------------

_CYCLE_CACHE = {}

def build_serpentine_cycle_from_anchor(W, H, anchor_top=True):
    rows = range(H) if anchor_top else range(H-1, -1, -1)
    order = []
    for idx, y in enumerate(rows):
        row = list(range(W))
        if idx % 2 == 1:
            row.reverse()
        for x in row:
            order.append((x, y))
    nxt = {}
    for i,(x,y) in enumerate(order):
        nx,ny = order[(i+1) % len(order)]
        nxt[(x,y)] = (nx,ny)
    return nxt

def get_cycle_map_from_side(board, head):
    H, W = len(board), len(board[0])
    anchor_top = (head[1] <= H//2)
    key = (W, H, anchor_top)
    if key not in _CYCLE_CACHE:
        _CYCLE_CACHE[key] = build_serpentine_cycle_from_anchor(W, H, anchor_top=anchor_top)
    return _CYCLE_CACHE[key]

def cycle_next_dir(nxt_map, x, y):
    nx, ny = nxt_map[(x,y)]
    dx, dy = nx - x, ny - y
    for d,(ox,oy) in DIRS.items():
        if (ox,oy) == (dx,dy):
            return d
    return "RIGHT"

# ----------------------------
# Contested scoring & cut/neck
# ----------------------------

def score_move_voronoi_min(board, our_head_next, opp_head):
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
    if not open_cell(board, nx, ny): return 0.0
    base = flood_component_size_xy(board, opp_head[0], opp_head[1])
    if base == 0: return 0.0
    saved = board[ny][nx]; board[ny][nx] = 1
    after = flood_component_size_xy(board, opp_head[0], opp_head[1])
    board[ny][nx] = saved
    gain = base - after
    near_deg = degree(board, opp_head[0], opp_head[1])
    choke = 1.0 if near_deg <= 2 else 0.0
    return (3.0 * gain + 5.0 * choke) if gain > 0 else (1.5 * choke)

# ----------------------------
# Seal (frontier necks)
# ----------------------------

def frontier_necks(board, head, opp_head, radius=4):
    best = []
    base = flood_component_size_xy(board, opp_head[0], opp_head[1])
    if base == 0: return best
    q = deque([(head[0], head[1], 0)])
    seen = {(head[0], head[1])}
    while q:
        x,y,d = q.popleft()
        if d >= radius: continue
        for dx,dy in DIRS.values():
            nx,ny = x+dx, y+dy
            if not open_cell(board,nx,ny) or (nx,ny) in seen: continue
            seen.add((nx,ny)); q.append((nx,ny,d+1))
            saved = board[ny][nx]; board[ny][nx] = 1
            after = flood_component_size_xy(board, opp_head[0], opp_head[1])
            board[ny][nx] = saved
            gain = base - after
            if gain > 0:
                best.append(((nx,ny), gain, d+1))
    best.sort(key=lambda t: (t[1] / (1 + t[2])), reverse=True)
    return best[:6]

def try_build_seal_plan(board, head, opp_head, my_boosts, turn_count):
    cands = frontier_necks(board, head, opp_head, radius=4)
    if not cands: return deque()
    our_dist = compute_distance_map(board, head, max_expansions=400)
    opp_dist = compute_distance_map(board, opp_head, max_expansions=400)
    for (cx,cy), gain, dist_est in cands:
        du = our_dist[cy][cx]; dv = opp_dist[cy][cx]
        if du == INF: continue
        path_dirs = shortest_path_dirs(board, head, (cx,cy))
        if not path_dirs: continue
        first_dir = path_dirs[0]
        win_now   = du < dv
        tie_now   = (du == dv and my_boosts > 0 and path_clear(head, first_dir, board, 3))
        win_later = (du == dv + 1 and my_boosts > 0 and path_clear(head, first_dir, board, 3))
        if not (win_now or tie_now or win_later): continue
        plan_len = max(3, min(12, len(path_dirs)))
        plan = deque(path_dirs[:plan_len])
        nx, ny = next_pos_xy(*head, plan[0])
        if not open_cell(board, nx, ny): continue
        return plan
    return deque()

# ----------------------------
# Opponent prediction & LOCAL sequential-judge guard
# ----------------------------

def infer_heading(trail):
    if len(trail) < 2: return "RIGHT"
    (x2,y2) = trail[-1]; (x1,y1) = trail[-2]
    dx,dy = x2-x1, y2-y1
    if   dx == 1: return "RIGHT"
    elif dx == -1: return "LEFT"
    elif dy == 1: return "DOWN"
    elif dy == -1: return "UP"
    return "RIGHT"

def likely_opponent_next_cells(board, opp_head, opp_heading):
    order = [opp_heading, LEFT_OF[opp_heading], RIGHT_OF[opp_heading]]
    cells = []
    for d in order:
        nx, ny = next_pos_xy(*opp_head, d)
        if open_cell(board, nx, ny):
            cells.append(((nx, ny), d))
    return cells

def sequential_guard_penalty(board, head, opp_head, opp_heading, cand_dir, player_num):
    """Localize guard: strong ≤2, medium at 3, mild 4–5, none beyond."""
    if opp_head is None:
        return 0.0
    dheads = manhattan(head, opp_head)
    if dheads > 5:
        return 0.0
    likely = likely_opponent_next_cells(board, opp_head, opp_heading)
    if not likely:
        return 0.0

    nx, ny = next_pos_xy(*head, cand_dir)

    if dheads <= 2:
        strong = 2000.0 if player_num == 2 else 1200.0
        medium = 800.0 if player_num == 2 else 500.0
    elif dheads == 3:
        strong = 1000.0 if player_num == 2 else 700.0
        medium = 400.0 if player_num == 2 else 250.0
    else:  # 4–5
        strong = 400.0
        medium = 150.0

    for (ox,oy), od in likely:
        if (nx,ny) == (ox,oy):
            return -strong

    for (ox,oy), od in likely:
        if (nx,ny) == opp_head:
            if next_pos_xy(*opp_head, od) == head:
                return -strong
            return -medium

    if player_num == 2 and manhattan((nx,ny), opp_head) == 0:
        return -strong
    if player_num == 2 and manhattan((nx,ny), opp_head) == 1:
        return -medium
    return 0.0

# ----------------------------
# Smarter opening
# ----------------------------

def choose_opening_dir(board, head, current_dir):
    """First move: steer toward midline (UP if bottom half, DOWN if top half) if safe."""
    H, W = len(board), len(board[0])
    x, y = head
    prefer = "UP" if y > H // 2 else "DOWN"
    order = [prefer, current_dir, LEFT_OF[current_dir], RIGHT_OF[current_dir], OPPOSITE[current_dir]]
    for d in order:
        nx, ny = next_pos_xy(x, y, d)
        if open_cell(board, nx, ny):
            return d
    return current_dir

# ----------------------------
# Decision
# ----------------------------

def decide_move(my_trail, other_trail, turn_count, my_boosts, player_num):
    global SEAL_MODE, SEAL_PLAN, SEAL_TARGET, SEAL_BLOCKS, SEAL_REPLAN_ONCE

    board = game_state.get("board")
    if board is None: return "RIGHT"

    head = my_trail[-1] if my_trail else (0,0)
    opp_head = other_trail[-1] if other_trail else None
    current_dir = infer_heading(my_trail)

    # Opening nudge toward midline (first couple of moves)
    if len(my_trail) < 3:
        od = choose_opening_dir(board, head, current_dir)
        # Only apply if not guarded-dangerous
        opp_heading = infer_heading(other_trail) if opp_head else "RIGHT"
        if sequential_guard_penalty(board, head, opp_head, opp_heading, od, player_num) >= 0:
            return od

    # legal candidates
    all_legal = [d for d in ORDERED_DIRS if open_cell(board, *next_pos_xy(*head, d))]
    if not all_legal:
        return current_dir
    directions = list(all_legal)
    opp_of_current = OPPOSITE.get(current_dir)
    if opp_of_current in directions and len(directions) > 1:
        directions.remove(opp_of_current)

    # Phases
    EARLY = turn_count < 30
    opp_heading = infer_heading(other_trail) if opp_head else "RIGHT"
    close = (opp_head is not None and manhattan(head, opp_head) <= 6)
    connected = (opp_head is not None and heads_connected(board, head, opp_head))
    SOLO = (opp_head is not None and not connected)

    # SOLO: anchored serpentine
    if SOLO:
        SEAL_MODE = False; SEAL_PLAN.clear(); SEAL_BLOCKS = 0; SEAL_REPLAN_ONCE = False
        nxt_map = get_cycle_map_from_side(board, head)
        d_cycle = cycle_next_dir(nxt_map, head[0], head[1])
        nx, ny = next_pos_xy(*head, d_cycle)
        if open_cell(board, nx, ny):
            allow_boost = (my_boosts > 0 and 30 <= turn_count <= 80 and path_clear(head, d_cycle, board, steps=3))
            return f"{d_cycle}:BOOST" if allow_boost else d_cycle

    # Follow SEAL plan if any
    if SEAL_MODE and SEAL_PLAN:
        d = SEAL_PLAN[0]
        nx, ny = next_pos_xy(*head, d)
        if open_cell(board, nx, ny):
            # guard even on plan
            if sequential_guard_penalty(board, head, opp_head, opp_heading, d, player_num) >= 0:
                SEAL_PLAN.popleft(); SEAL_BLOCKS = 0
                allow_boost = (my_boosts > 0 and path_clear(head, d, board, steps=3)
                               and 20 <= turn_count <= 80 and tunnel_len(board, head[0], head[1], d) == 0)
                if not SEAL_PLAN:
                    SEAL_MODE = False; SEAL_REPLAN_ONCE = False
                return f"{d}:BOOST" if allow_boost else d
        else:
            SEAL_BLOCKS += 1
            if SEAL_BLOCKS >= 2:
                SEAL_MODE = False; SEAL_PLAN.clear(); SEAL_REPLAN_ONCE = False
            else:
                if not SEAL_REPLAN_ONCE and SEAL_TARGET:
                    new_dirs = shortest_path_dirs(board, head, SEAL_TARGET)
                    if new_dirs:
                        SEAL_PLAN = deque(new_dirs[:max(3, min(12, len(new_dirs)))])
                        SEAL_REPLAN_ONCE = True
                        d = SEAL_PLAN[0]
                        nx, ny = next_pos_xy(*head, d)
                        if open_cell(board, nx, ny) and sequential_guard_penalty(board, head, opp_head, opp_heading, d, player_num) >= 0:
                            SEAL_PLAN.popleft()
                            allow_boost = (my_boosts > 0 and path_clear(head, d, board, steps=3)
                                           and 20 <= turn_count <= 80 and tunnel_len(board, head[0], head[1], d) == 0)
                            return f"{d}:BOOST" if allow_boost else d

    # Try start SEAL plan
    if connected and opp_head:
        new_plan = try_build_seal_plan(board, head, opp_head, my_boosts, turn_count)
        if new_plan:
            d0 = new_plan[0]
            if sequential_guard_penalty(board, head, opp_head, opp_heading, d0, player_num) >= 0:
                SEAL_MODE = True; SEAL_PLAN = new_plan
                SEAL_TARGET = None; SEAL_BLOCKS = 0; SEAL_REPLAN_ONCE = False
                d = SEAL_PLAN.popleft()
                allow_boost = (my_boosts > 0 and path_clear(head, d, board, steps=3)
                               and (EARLY or (30 <= turn_count <= 80))
                               and tunnel_len(board, head[0], head[1], d) == 0)
                return f"{d}:BOOST" if allow_boost else d

    # Normal scoring + guard + axis alternation + center pull
    scored = []
    best_tuple = None
    best_dir = directions[0]
    axis_bias = recent_axis_bias(my_trail, k=4)

    for d in directions:
        nx, ny = next_pos_xy(*head, d)
        if not open_cell(board, nx, ny): continue

        if opp_head and connected and close:
            primary = score_move_voronoi_min(board, (nx, ny), opp_head)
            primary += cut_bonus(board, opp_head, nx, ny)
        elif EARLY and not close:
            primary = area_score(nx, ny, board, limit=70)
            if wall_distance(board, nx, ny) >= 2: primary += 1.0
        else:
            primary = area_score(nx, ny, board, limit=50)
            if not connected and wall_distance(board, nx, ny) >= 2: primary += 0.5

        deg1 = degree(board, nx, ny)
        fx, fy = next_pos_xy(nx, ny, d)
        deg2 = degree(board, fx, fy) if open_cell(board, fx, fy) else 0
        primary += 0.2 * (deg1 + deg2)

        tlen = tunnel_len(board, head[0], head[1], d, maxn=6)
        primary -= 0.5 * tlen

        wd = wall_distance(board, nx, ny)
        if wd <= 1:
            primary -= 0.8

        # Stronger early center pull
        if turn_count < 20:
            primary += 0.8 * center_bias_delta(board, head[0], head[1], nx, ny)
        else:
            primary += 0.3 * center_bias_delta(board, head[0], head[1], nx, ny)

        # Axis alternation
        if axis_bias == "H" and d in ("UP","DOWN"):
            primary += 0.6
        elif axis_bias == "V" and d in ("LEFT","RIGHT"):
            primary += 0.6

        # Local sequential-judge guard
        guard = sequential_guard_penalty(board, head, opp_head, opp_heading, d, player_num)
        primary += guard  # negative when risky

        straight = 1 if d == current_dir else 0
        inward = center_bias_delta(board, head[0], head[1], nx, ny)
        tie = (primary, straight, inward, deg1, (deg1 + deg2), wd)
        scored.append((tie, d))
        if best_tuple is None or tie > best_tuple:
            best_tuple = tie; best_dir = d

    # Relative tie-breaker
    equal_dirs = [d for (t,d) in scored if t == best_tuple]
    if len(equal_dirs) > 1:
        best_dir = min(equal_dirs, key=lambda d: rel_order(current_dir, d))

    # Boost policy (safe)
    nx, ny = next_pos_xy(*head, best_dir)
    allow_boost = (
        my_boosts > 0
        and path_clear(head, best_dir, board, steps=3)
        and tunnel_len(board, head[0], head[1], best_dir) == 0
        and sequential_guard_penalty(board, head, opp_head, opp_heading, best_dir, player_num) >= 0
    )
    if turn_count < 30:
        n1x, n1y = nx, ny
        n2x, n2y = next_pos_xy(n1x, n1y, best_dir)
        good_curve = center_bias_delta(board, head[0], head[1], n2x, n2y) or degree(board, n2x, n2y) >= 3
        allow_boost = allow_boost and good_curve
    else:
        allow_boost = allow_boost and (best_tuple[0] > 5) and (30 <= turn_count <= 80)

    return f"{best_dir}:BOOST" if allow_boost else best_dir

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

    move = decide_move(my_trail, other_trail, turn_count, my_boosts, player_number)
    return jsonify({"move": move}), 200

@app.route("/end", methods=["POST"])
def end_game():
    global SEAL_MODE, SEAL_PLAN, SEAL_TARGET, SEAL_BLOCKS, SEAL_REPLAN_ONCE
    SEAL_MODE = False; SEAL_PLAN.clear(); SEAL_TARGET = None
    SEAL_BLOCKS = 0; SEAL_REPLAN_ONCE = False
    data = request.get_json()
    if data:
        result = data.get("result", "UNKNOWN")
        print(f"\nGame Over! Result: {result}")
    return jsonify({"status": "acknowledged"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
