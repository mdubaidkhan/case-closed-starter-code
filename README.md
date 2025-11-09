# 🕵️‍♂️ Case Closed AI Agent — Hybrid Weighted-Centroid Strategy

This repository contains my final submission for the **Case Closed** AI Programming Challenge.

**Challenge Theme:**  
Design an intelligent autonomous agent that can survive longer than its opponent in a grid-based environment inspired by *Tron Lightcycle* — where every move leaves a permanent trail, and collisions mean game over.

---

## ⚙️ Overview

The agent (`agent.py`) runs a lightweight **Flask** server that receives JSON game states and responds with optimal movement decisions (`UP`, `DOWN`, `LEFT`, or `RIGHT`).

It uses a blend of **spatial heuristics**, **Voronoi territory estimation**, and **weighted centroid logic** to maximize survivability and territory control.  
No machine learning libraries are used — the logic is purely algorithmic, ensuring fast response times.

---

## 🧠 Core Algorithm Features

### 🔹 Phase-Adaptive Decision Logic
The agent dynamically switches between multiple strategies depending on the game state:
- **Early Game:** Expands safely and avoids corners.
- **Contested Mid-Game:** Uses Voronoi territory scoring to control open regions.
- **Seal Mode:** Detects chokepoints to trap the opponent.
- **Solo Mode:** Switches to serpentine traversal once both regions are disconnected.

### 🔹 Weighted-Centroid Movement
Each turn, the agent calculates the centroid of reachable open cells — **weighted by accessibility, opponent distance, and wall proximity** — and moves toward it, ensuring balanced territory coverage and avoiding directional bias.

### 🔹 Guard & Cut Logic
Predicts the opponent’s likely next moves and avoids high-risk paths.
It can identify neck regions that would isolate the opponent and opportunistically seal them off.

### 🔹 Boost Safety
When boosts are available, the agent performs multi-step lookahead to confirm that the boost path is safe and advantageous before activation.

---

