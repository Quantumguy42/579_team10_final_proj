"""
navigation4.py — FOODIE Multi-Robot Delivery Simulation
========================================================
Simulates a fleet of delivery robots navigating a weighted grid graph.

Pipeline:
  1. Orders are submitted with metadata (size, frozen, fragile).
  2. "Assign Bot" clusters unassigned orders via K-Medoids, then plans
     each cluster's route with Nearest-Neighbour TSP + 2-Opt refinement.
  3. Each robot runs a step-by-step FSM (Finite State Machine) that
     navigates the planned route using A* pathfinding, dynamically
     recalculating if a graph edge is removed mid-delivery.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue
import random
from enum import Enum, auto


import foodie_bagger as gb


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

WAREHOUSE_LOC = (1, 2)          # Grid coordinate of the warehouse / home base
NUM_BOTS      = 3               # Number of delivery robots
BOT_COLORS    = ["#9270A7", "#C559BE", "#BA4D5F", "yellow"]  # One colour per bot


# ──────────────────────────────────────────────────────────────────────────────
# Robot FSM States
# ──────────────────────────────────────────────────────────────────────────────

class State(Enum):
    AT_WAREHOUSE    = auto()   # Idle at warehouse, waiting for orders
    PICKUP          = auto()   # Orders loaded — transition to path calculation
    CALCULATE_PATHS = auto()   # Run A* for each destination in sequence
    STEP_THROUGH    = auto()   # Move one node along the planned path
    DROP_OFF        = auto()   # Arrived at a delivery node; remove order


# ──────────────────────────────────────────────────────────────────────────────
# Robot
# ──────────────────────────────────────────────────────────────────────────────

class Robot:
    """
    Represents a single delivery robot.

    Each robot maintains:
      - Its current grid location.
      - An ordered list of destination dicts  {"dst", "size", "frozen", "fragile"}.
      - A planned path (list of grid nodes) and a pointer into that path.
      - An FSM state that drives one step of behaviour per call to
        run_algorithm_fsm().
    """

    def __init__(self, name: str, map: nx.Graph):
        self.name     = name
        self.map      = map                 # Shared weighted graph
        self.location = WAREHOUSE_LOC       # Current position on the grid

        self.delivering_orders = False      # True once begin_delivering() is called

        self.bagging_results = ""           # For printing the foodie_bagger results

        self.path_pos    = 0                # Index of current position in self.path
        self.path        = [self.location]  # Planned route as a list of grid nodes
        self.destinations = []              # Ordered list of order dicts to visit

        self.bot_state = State.AT_WAREHOUSE

    # ── Pathfinding helpers ───────────────────────────────────────────────────

    def heuristic(self, a: tuple, b: tuple) -> int:
        """Manhattan distance heuristic for A*."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _cost_to_loc(self, dst: tuple) -> float:
        """Return the A* path cost from the current location to dst."""
        return nx.astar_path_length(
            self.map, self.location, dst,
            heuristic=self.heuristic, weight='weight'
        )

    def _find_path(self, src: tuple, dst: tuple):
        """
        Return (path, cost) from src to dst using A*.
        Shows a messagebox and returns None on failure.
        """
        try:
            path = nx.astar_path(
                self.map, src, dst,
                heuristic=self.heuristic, weight='weight'
            )
            cost = nx.astar_path_length(
                self.map, src, dst,
                heuristic=self.heuristic, weight='weight'
            )
            return path, cost
        except Exception as e:
            messagebox.showerror("Pathfinding Error", str(e))
            return None, None

    # ── Public API ────────────────────────────────────────────────────────────

    def set_destinations(self, route: list):
        """Replace the destination list with a new ordered route of order dicts."""
        self.destinations = route

    def recieve_order(self, order: dict):
        """Append a single order dict to the destination list."""
        self.destinations.append(order)

    def begin_delivering(self):
        """
        Arm the robot for delivery.
        Has no effect if there are no destinations loaded.
        """
        if len(self.destinations) > 0:
            self.delivering_orders = True

    # ── FSM ───────────────────────────────────────────────────────────────────

    def run_algorithm_fsm(self):
        """
        Advance the robot by one FSM tick.

        State transitions
        -----------------
        AT_WAREHOUSE   → PICKUP            (when delivering_orders and destinations exist)
        PICKUP         → CALCULATE_PATHS   (unconditional)
        CALCULATE_PATHS→ STEP_THROUGH      (after A* plans the full route)
        STEP_THROUGH   → STEP_THROUGH      (still travelling between stops)
                       → DROP_OFF          (arrived at a delivery node)
                       → AT_WAREHOUSE      (arrived back at warehouse)
                       → CALCULATE_PATHS   (next edge removed — replan)
        DROP_OFF       → STEP_THROUGH      (more stops remain)
                       → CALCULATE_PATHS   (all stops done; head home)
        """

        # ── Waiting at warehouse ──────────────────────────────────────────────
        if self.bot_state == State.AT_WAREHOUSE:
            if self.delivering_orders and len(self.destinations) > 0:
                self.bot_state = State.PICKUP

        # ── Orders loaded — move to planning ─────────────────────────────────
        elif self.bot_state == State.PICKUP:
            
            items = []

            for order in self.destinations:
                item = gb.Item()

                item.name = order["name"]
                item.size = order["size"]
                item.type = "frozen" if order["frozen"] else "regular"
                item.fragile = order["fragile"]

                items.append(item)
            
            
            # Run bagging
            gb.processAllItems(items)

            # Read results
            bags = gb.getBags()



            bag_text = ""

            for i, bag in enumerate(bags):

                bag_text += f"Bag {i+1} ({bag.type})\n"

                for item in bag.items:
                    bag_text += f"  - {item.name}\n"

                bag_text += "\n"

            self.bagging_results = bag_text



            self.bot_state = State.CALCULATE_PATHS

        # ── Plan the full route with A* ───────────────────────────────────────
        elif self.bot_state == State.CALCULATE_PATHS:
            src = self.location
            # Keep the already-travelled prefix so path_pos stays valid
            self.path = self.path[:self.path_pos + 1]

            for order in self.destinations:
                dst_coord = order["dst"]
                segment   = nx.astar_path(
                    self.map, src, dst_coord,
                    heuristic=self.heuristic, weight='weight'
                )
                self.path += segment[1:]   # drop the duplicate start node
                src = dst_coord

            print(f"[{self.name}] New path: {self.path}")
            self.bot_state = State.STEP_THROUGH

        # ── Move one node along the path ──────────────────────────────────────
        elif self.bot_state == State.STEP_THROUGH:
            # Guard: nothing left to traverse
            if self.path_pos + 1 >= len(self.path):
                self.bot_state = State.CALCULATE_PATHS
                return

            next_node = self.path[self.path_pos + 1]
            edge_exists = self.map.has_edge(self.location, next_node)
            print(f"[{self.name}] {self.location} → {next_node}: edge={edge_exists}")

            if edge_exists:
                # Advance position
                self.path_pos += 1
                self.location  = self.path[self.path_pos]
                print(f"[{self.name}] Now at {self.location}")

                # Check whether we've arrived at a destination
                at_destination = any(
                    d["dst"] == self.location for d in self.destinations
                )

                if at_destination:
                    if self.location == WAREHOUSE_LOC:
                        # Returned home — reset everything
                        self.delivering_orders = False
                        self.path     = [self.location]
                        self.path_pos = 0
                        self.bagging_results = ""
                        # Remove the warehouse entry from the destination list
                        idx = next(
                            i for i, d in enumerate(self.destinations)
                            if d["dst"] == self.location
                        )
                        self.destinations.pop(idx)
                        self.bot_state = State.AT_WAREHOUSE
                    else:
                        self.bot_state = State.DROP_OFF
                else:
                    self.bot_state = State.STEP_THROUGH   # Keep moving
            else:
                # Edge was removed — replan from current position
                self.bot_state = State.CALCULATE_PATHS

        # ── Deliver the package at the current node ───────────────────────────
        elif self.bot_state == State.DROP_OFF:
            # Remove the delivered order from the list
            idx = next(
                i for i, d in enumerate(self.destinations)
                if d["dst"] == self.location
            )
            self.destinations.pop(idx)

            if len(self.destinations) > 0:
                # More stops — keep going
                self.bot_state = State.STEP_THROUGH
            else:
                # All deliveries done — navigate back to warehouse
                self.destinations = [{
                    "dst":     WAREHOUSE_LOC,
                    "size":    "N/A",
                    "frozen":  False,
                    "fragile": False,
                }]
                self.bot_state = State.CALCULATE_PATHS


# ──────────────────────────────────────────────────────────────────────────────
# FoodieApp — Tkinter GUI
# ──────────────────────────────────────────────────────────────────────────────

class FoodieApp:
    """
    Main application window.

    Responsibilities
    ----------------
    - Render the delivery grid (matplotlib embedded in Tk).
    - Accept new orders with metadata via the control panel.
    - Cluster and assign orders to available robots (assign_bot).
    - Step each robot's FSM individually via the "Step" buttons.
    - Allow interactive edge add/remove via canvas clicks.
    """

    def __init__(self, root: tk.Tk, G: nx.Graph, node_types: dict):
        self.root       = root
        self.root.title("FOODIE Delivery Grid")
        self.G          = G
        self.node_types = node_types   # node → label string (e.g. "H2", "S1", "I")

        self.current_path = []         # Path being highlighted on the graph
        self.assigned_bot = None       # Last bot assigned (used for highlight)
        self.clusters     = []         # List of clusters (each a list of order dicts)

        # ── Order-form state variables ────────────────────────────────────────
        self.order_size    = tk.StringVar(value="medium")
        self.order_frozen  = tk.StringVar(value="no")
        self.order_fragile = tk.StringVar(value="no")

        # ── Robots ───────────────────────────────────────────────────────────
        self.bots   = [Robot(f"B{i}", self.G) for i in range(NUM_BOTS)]
        self.groups = [[] for _ in range(NUM_BOTS)]

        # ── Pre-populated debug orders ────────────────────────────────────────
        self.unassigned_destinations = [
            {"dst": (0, 0), "name": "beat", "size": "small",  "frozen": False, "fragile": True},
            {"dst": (0, 9), "name": "grape", "size": "medium", "frozen": True,  "fragile": False},
            {"dst": (1, 5), "name": "onion", "size": "large",  "frozen": False, "fragile": False},
            {"dst": (3, 8), "name": "pinapple", "size": "small",  "frozen": True,  "fragile": True},
            {"dst": (5, 1), "name": "watermelon", "size": "medium", "frozen": False, "fragile": False},
            {"dst": (5, 9), "name": "orange", "size": "large",  "frozen": True,  "fragile": False},
            {"dst": (7, 3), "name": "squash", "size": "small",  "frozen": False, "fragile": True},
            {"dst": (9, 0), "name": "salt", "size": "medium", "frozen": False, "fragile": False},
            {"dst": (9, 9), "name": "pepper", "size": "large",  "frozen": True,  "fragile": True},
        ]

        self.current_order = [None, None]

        # Edge-editing mode: "none" | "add" | "remove"
        self.edit_mode     = tk.StringVar(value="none")
        self.selected_node = None      # First node clicked in edge-edit mode

        self._build_ui()
        self.draw_graph()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        """Build the left control panel and right graph canvas."""



        bag_panel = tk.Frame(self.root, padx=10, pady=10)
        bag_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.bagging_var       = tk.StringVar(value="")

        tk.Label(
            bag_panel,
            text="Bagging Results",
            font=("Arial", 13, "bold")
        ).pack(anchor="w")

        # self.bagging_text = tk.Text(
        #     bag_panel,
        #     width=30,
        #     height=35,
        #     wrap="word"
        # )

        tk.Label(bag_panel, textvariable=self.bagging_var,
                 wraplength=160, justify="left", fg="black").pack(anchor="w")

        # ── Left panel ───────────────────────────────────────────────────────
        # ── Scrollable Left Panel ─────────────────────────────

        left_container = tk.Frame(self.root)
        left_container.pack(side=tk.LEFT, fill=tk.Y)

        scrollbar = tk.Scrollbar(left_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas = tk.Canvas(
            left_container,
            width=250,
            yscrollcommand=scrollbar.set
        )

        canvas.pack(side=tk.LEFT, fill=tk.Y)

        scrollbar.config(command=canvas.yview)

        # Actual control frame INSIDE canvas
        ctrl = tk.Frame(canvas, padx=10, pady=10)

        canvas.create_window(
            (0, 0),
            window=ctrl,
            anchor="nw"
        )

        # Auto-update scroll region
        ctrl.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )



        # Order creation form
        tk.Label(ctrl, text="Create Order", font=("Arial", 13, "bold")).pack(anchor="w")

        tk.Label(ctrl, text='Item Name:').pack(anchor="w")
        self.item_entry = tk.Entry(ctrl, width=12)
        self.item_entry.insert(0, "tomatoe")
        self.item_entry.pack(anchor="w", pady=2)

        tk.Label(ctrl, text='Destination "row,col":').pack(anchor="w")
        self.dst_entry = tk.Entry(ctrl, width=12)
        self.dst_entry.insert(0, "1,3")
        self.dst_entry.pack(anchor="w", pady=2)

        # Size radio buttons
        tk.Label(ctrl, text="Size:").pack(anchor="w", pady=(6, 0))
        size_frame = tk.Frame(ctrl)
        size_frame.pack(anchor="w")
        for val, label in [("small", "Small"), ("medium", "Medium"), ("large", "Large")]:
            tk.Radiobutton(size_frame, text=label,
                           variable=self.order_size, value=val).pack(side="left")

        # Frozen radio buttons
        tk.Label(ctrl, text="Frozen:").pack(anchor="w", pady=(6, 0))
        frozen_frame = tk.Frame(ctrl)
        frozen_frame.pack(anchor="w")
        for val, label in [("no", "No"), ("yes", "Yes")]:
            tk.Radiobutton(frozen_frame, text=label,
                           variable=self.order_frozen, value=val).pack(side="left")

        # Fragile radio buttons
        tk.Label(ctrl, text="Fragile:").pack(anchor="w", pady=(6, 0))
        fragile_frame = tk.Frame(ctrl)
        fragile_frame.pack(anchor="w")
        for val, label in [("no", "No"), ("yes", "Yes")]:
            tk.Radiobutton(fragile_frame, text=label,
                           variable=self.order_fragile, value=val).pack(side="left")

        tk.Button(ctrl, text="Submit Order", command=self.add_order_to_queue,
                  bg="#4C63AF", fg="white", width=14).pack(pady=6)
        tk.Button(ctrl, text="Clear Unassigned", command=self.clear_unassigned_orders,
                  bg="#8B8C8D", fg="white", width=14).pack(pady=6)

        self.assign_bot_btn = tk.Button(ctrl, text="Assign Bot", command=self.assign_bot,
                                        bg="#4CAF4E", fg="white", width=14)
        self.assign_bot_btn.pack(pady=6)

        # Unassigned order list display
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=12)
        tk.Label(ctrl, text="New Orders", font=("Arial", 13, "bold")).pack(anchor="w")
        self.order_list = tk.StringVar(value="No orders.")
        tk.Label(ctrl, textvariable=self.order_list,
                 wraplength=160, justify="left", fg="black").pack(anchor="w")

        # Per-robot controls
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=12)
        tk.Label(ctrl, text="Robot Control", font=("Arial", 13, "bold")).pack(anchor="w")

        self.bot_buttons       = []
        self.bot_start_buttons = []
        self.bot_states        = []
        self.bot_orders        = []

        for i in range(NUM_BOTS):
            # Row 1: label + Step + Start Delivery buttons
            row1 = tk.Frame(ctrl)
            row1.pack(fill="x", pady=5)
            tk.Label(row1, text=f"Bot {i}", font=("Arial", 13, "bold")).pack(side="left", padx=5)

            state_frame = tk.Frame(ctrl)
            state_lbl   = tk.Label(state_frame, text="State: ", font=("Arial", 10))
            self.bot_states.append(state_lbl)

            bot_order_frame = tk.Frame(ctrl)
            order_var       = tk.StringVar(value="No orders.")
            self.bot_orders.append(order_var)
            tk.Label(bot_order_frame, textvariable=order_var,
                     wraplength=160, justify="left", fg="black").pack(side="left", padx=5)

            # Start Delivery button — needs its own handle for the lambda closure
            start_btn = tk.Button(row1, text="Start Delivery", bg=BOT_COLORS[i])
            start_btn.config(
                command=lambda bot=self.bots[i], btn=start_btn:
                    self.start_delivery(bot, btn)
            )
            self.bot_start_buttons.append(start_btn)

            # Step button
            step_btn = tk.Button(row1, text="Step", bg="#4CAF4E", fg="white", width=14)
            step_btn.config(
                command=lambda bot=self.bots[i], lbl=state_lbl, sbtn=start_btn:
                    self.algo_step(bot, lbl, sbtn)
            )
            self.bot_buttons.append(step_btn)

            step_btn.pack(side="left", padx=5)
            start_btn.pack(side="left", padx=5)

            # Row 2: state label
            state_frame.pack(fill="x", pady=(0, 4))
            state_lbl.pack(side="left", padx=5)

            # Row 3: order list for this bot
            bot_order_frame.pack(fill="x", pady=(0, 8))

            # Initialise state label text
            self.bot_states[i].config(text=f"State: {self.bots[i].bot_state}")

        # Edge weight entry (used when adding edges via canvas click)
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=12)
        tk.Label(ctrl, text="Weight:").pack(anchor="w")
        self.edge_weight_entry = tk.Entry(ctrl, width=12)
        self.edge_weight_entry.insert(0, "1")
        self.edge_weight_entry.pack(anchor="w", pady=2)

        # Status bar
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=12)
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(ctrl, textvariable=self.status_var,
                 wraplength=160, justify="left", fg="gray").pack(anchor="w")

        # ── Right panel: graph canvas ─────────────────────────────────────────
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Edge editing mode selector (below canvas, packed into ctrl)
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=12)
        tk.Label(ctrl, text="Edit Edges", font=("Arial", 13, "bold")).pack(anchor="w")
        for mode, label in [("none", "None"), ("add", "Add Edge"), ("remove", "Remove Edge")]:
            tk.Radiobutton(ctrl, text=label,
                           variable=self.edit_mode, value=mode).pack(side="left")

        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self._update_order_lists()

    # ── Canvas interaction ────────────────────────────────────────────────────

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(
            int(-1 * (event.delta / 120)),
            "units"
        )

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_canvas_click(self, event):
        """
        Handle a click on the matplotlib canvas.

        In "add" or "remove" mode:
          - First click selects a node (highlights it cyan).
          - Second click completes the edge operation between the two nodes.
          - Clicking the same node twice, or clicking outside a node, cancels.
        """
        if event.inaxes != self.ax or self.edit_mode.get() == "none":
            return

        pos = {(r, c): (c, -r) for r, c in self.G.nodes()}

        # Find the nearest node to the click
        clicked_node = min(
            self.G.nodes(),
            key=lambda n: (pos[n][0] - event.xdata) ** 2
                        + (pos[n][1] - event.ydata) ** 2
        )

        # Reject clicks that are too far from any node
        nx_, ny_ = pos[clicked_node]
        dist = ((nx_ - event.xdata) ** 2 + (ny_ - event.ydata) ** 2) ** 0.5
        if dist > 0.5:
            self.selected_node = None
            self.draw_graph()
            return

        if self.selected_node is None:
            # First click: select node and highlight it
            self.selected_node = clicked_node
            self.status_var.set(f"Selected: {clicked_node}. Click another node.")
            self.draw_graph()
            x, y = pos[clicked_node]
            self.ax.plot(x, y, "o", ms=20, color="cyan", alpha=0.5, zorder=5)
            self.canvas.draw()
        else:
            # Second click: perform the edge operation
            u, v = self.selected_node, clicked_node
            if u != v:
                if self.edit_mode.get() == "add":
                    w = int(self.edge_weight_entry.get() or 1)
                    self.G.add_edge(u, v, weight=w)
                    self.status_var.set(f"Added edge {u} ↔ {v} (w={w})")
                elif self.edit_mode.get() == "remove":
                    if self.G.has_edge(u, v):
                        self.G.remove_edge(u, v)
                        self.status_var.set(f"Removed edge {u} ↔ {v}")
                    else:
                        self.status_var.set(f"No edge between {u} and {v}")

            self.selected_node = None
            self.draw_graph()

    # ── Parsing helpers ───────────────────────────────────────────────────────

    def _parse_node(self, text: str) -> tuple:
        """Parse a "row,col" string into an (int, int) tuple."""
        r, c = map(int, text.strip().split(","))
        return (r, c)

    def _bot_at_loc(self, n: tuple) -> bool:
        """Return True if any robot is currently at node n."""
        return any(bot.location == n for bot in self.bots)

    # ── Graph colouring ───────────────────────────────────────────────────────

    def _node_colors(self) -> list:
        """
        Return a colour for every node in graph iteration order.

        Priority (highest first):
          1. Bot-destination node      → that bot's colour
          2. Assigned-bot location     → yellow
          3. Any bot location          → orange
          4. Default                   → light gray
        """
        colors = []
        for n in self.G.nodes():
            color = None
            # Check if any bot has this node as an upcoming delivery
            for i, bot in enumerate(self.bots):
                if any(d["dst"] == n for d in bot.destinations) \
                        and n != WAREHOUSE_LOC:
                    color = BOT_COLORS[i]
                    break

            if color is None:
                if self.assigned_bot is not None and n == self.assigned_bot.location:
                    color = "yellow"
                elif self._bot_at_loc(n):
                    color = "orange"
                else:
                    color = "lightgray"
            colors.append(color)
        return colors

    def _node_outline_colors(self) -> list:
        """
        Return an outline colour for each node based on cluster membership.
        Nodes not in any cluster get a black outline.
        """
        outline_colors = []
        for n in self.G.nodes():
            color = "black"
            for i, cluster in enumerate(self.clusters):
                # clusters[i] is a list of order dicts
                if any(d["dst"] == n for d in cluster if isinstance(d, dict)):
                    color = BOT_COLORS[i]
                    break
            outline_colors.append(color)
        return outline_colors

    def _node_outline_thicknesses(self) -> list:
        """Highlight source/destination nodes of the current order with a thick border."""
        return [2 if n in self.current_order else 0 for n in self.G.nodes()]

    def _edge_colors(self) -> list:
        """Colour path edges red, all others gray."""
        path_edges = set(zip(self.current_path, self.current_path[1:]))
        return [
            "red" if (u, v) in path_edges or (v, u) in path_edges else "#AAAAAA"
            for u, v in self.G.edges()
        ]

    def _edge_widths(self) -> list:
        """Make path edges thicker than background edges."""
        path_edges = set(zip(self.current_path, self.current_path[1:]))
        return [
            3.0 if (u, v) in path_edges or (v, u) in path_edges else 1.0
            for u, v in self.G.edges()
        ]

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_graph(self):
        """Redraw the entire grid graph with current node/edge colours and bot labels."""
        self.ax.clear()
        pos        = {(r, c): (c, -r) for r, c in self.G.nodes()}
        labels     = {n: self.node_types.get(n, "I") for n in self.G.nodes()}
        edge_labels = nx.get_edge_attributes(self.G, 'weight')

        nx.draw(
            self.G, pos, ax=self.ax,
            node_color=self._node_colors(),
            edgecolors=self._node_outline_colors(),
            linewidths=self._node_outline_thicknesses(),
            edge_color=self._edge_colors(),
            width=self._edge_widths(),
            node_size=1200,
            labels=labels,
            with_labels=True,
            font_weight='bold',
        )
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, ax=self.ax)

        # Annotate each bot's current position with its name
        for i, bot in enumerate(self.bots):
            x, y = pos[bot.location]
            self.ax.annotate(
                bot.name,
                xy=(x, y),
                xytext=(x + 0.3, y + 0.3),
                fontsize=9,
                color="darkred",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=BOT_COLORS[i], alpha=0.7),
            )

        self.ax.set_title("FOODIE Delivery Grid")
        self.canvas.draw()

    # ── Order management ──────────────────────────────────────────────────────

    def _update_order_lists(self):
        """Refresh the unassigned-order label and all per-bot order labels."""
        if self.unassigned_destinations:
            self.order_list.set(
                [d["dst"] for d in self.unassigned_destinations]
            )
        else:
            self.order_list.set("No Orders")

        for i, bot in enumerate(self.bots):
            if bot.destinations:
                self.bot_orders[i].set([d["dst"] for d in bot.destinations])
            else:
                self.bot_orders[i].set("No orders assigned to bot")

    def add_order_to_queue(self):
        """
        Parse the destination entry and add a new order dict to the unassigned list.
        Silently rejects orders that:
          - Target the warehouse itself.
          - Duplicate an existing unassigned order.
          - Duplicate a destination already on a bot.
        """
        try:
            dst = self._parse_node(self.dst_entry.get())


            if dst == WAREHOUSE_LOC:
                return

            existing = [
                o["dst"] if isinstance(o, dict) else o
                for o in self.unassigned_destinations
            ]
            if dst in existing:
                return
            if any(dst in [d["dst"] for d in bot.destinations] for bot in self.bots):
                return

            order = {
                "dst":     dst,
                "name":    self.item_entry.get().lower(),
                "size":    self.order_size.get(),
                "frozen":  self.order_frozen.get() == "yes",
                "fragile": self.order_fragile.get() == "yes",
            }
            self.unassigned_destinations.append(order)
            self._update_order_lists()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def clear_unassigned_orders(self):
        """Remove all orders from the unassigned queue."""
        self.unassigned_destinations = []
        self._update_order_lists()

    def start_delivery(self, bot: Robot, button_handle: tk.Button):
        """Arm the given robot for delivery and disable its Start button."""
        if len(bot.destinations) > 0:
            bot.begin_delivering()
            button_handle.config(state=tk.DISABLED)

    # ── Bot assignment ────────────────────────────────────────────────────────

    def heuristic(self, a: tuple, b: tuple) -> int:
        """Manhattan distance (used by gen_dist_matrix)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def gen_dist_matrix(self, source: tuple, destinations: list) -> dict:
        """
        Build a full pairwise distance matrix between source and all destinations.

        Returns a dict-of-dicts: dist_matrix[from_node][to_node] = A* cost.
        The source row is included so TSP can measure from the warehouse.
        """
        dist_matrix = {}
        all_locations = [source] + destinations
        for loc in all_locations:
            dist_matrix[loc] = {}
            for dest in destinations:
                dist_matrix[loc][dest] = nx.astar_path_length(
                    self.G, loc, dest,
                    heuristic=self.heuristic, weight='weight'
                )
        return dist_matrix

    def k_medoids(self, destinations: list, dist_matrix: dict,
                  k: int, max_iters: int = 100, seed: int = 0) -> list:
        """
        Partition destinations into k clusters using graph distances (K-Medoids).

        Unlike K-Means, medoids must be real nodes, making it robust on
        weighted / non-Euclidean graphs.

        Parameters
        ----------
        destinations : list of grid-coordinate tuples
        dist_matrix  : pairwise distance dict built by gen_dist_matrix
        k            : number of clusters (capped at len(destinations))
        max_iters    : iteration limit
        seed         : random seed for reproducibility

        Returns
        -------
        labels : list[int] — labels[i] is the cluster index for destinations[i]
        """
        k   = min(k, len(destinations))
        rng = random.Random(seed)

        medoid_indices = rng.sample(range(len(destinations)), k)
        labels         = [0] * len(destinations)

        for _ in range(max_iters):
            medoids = [destinations[i] for i in medoid_indices]

            # Assignment step: each point → nearest medoid
            new_labels = [
                min(range(k), key=lambda i: dist_matrix[dest][medoids[i]])
                for dest in destinations
            ]

            if new_labels == labels:
                break
            labels = new_labels

            # Update step: best medoid minimises total in-cluster distance
            for ci in range(k):
                cluster = [destinations[j] for j, l in enumerate(labels) if l == ci]
                if not cluster:
                    continue
                best = min(
                    cluster,
                    key=lambda c: sum(dist_matrix[c][p] for p in cluster)
                )
                medoid_indices[ci] = destinations.index(best)

        return labels

    def nearest_neighbor_tsp(self, cluster: list, start: tuple,
                              dist_matrix: dict) -> list:
        """
        Greedy Nearest-Neighbour TSP heuristic.

        Starting from `start`, repeatedly visit the closest unvisited node.
        Returns the ordered list of stops (not including start).
        """
        if not cluster:
            return []

        unvisited = list(cluster)
        route     = []
        current   = start

        while unvisited:
            nearest = min(unvisited, key=lambda p: dist_matrix[current][p])
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        return route

    def two_opt(self, route: list, start: tuple, dist_matrix: dict) -> list:
        """
        2-Opt local search to improve a TSP route.

        Repeatedly reverses segments of the route when doing so reduces the
        total path cost, until no improvement is possible.

        Works on graph distances (not Euclidean), so it is correct for
        weighted grids where the shortest path between two nodes may not
        be a straight line.

        Returns the improved route (start node is dropped from the output).
        """
        if len(route) < 4:
            return route

        path     = [start] + list(route)
        improved = True

        while improved:
            improved = False
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path) - 1):
                    before = (dist_matrix[path[i - 1]][path[i]]
                            + dist_matrix[path[j]][path[j + 1]])
                    after  = (dist_matrix[path[i - 1]][path[j]]
                            + dist_matrix[path[i]][path[j + 1]])
                    if after < before - 1e-10:
                        path[i:j + 1] = reversed(path[i:j + 1])
                        improved = True

        return path[1:]   # drop the prepended start node

    def assign_bot(self):
        """
        Cluster all pending orders and assign optimised routes to available robots.

        Steps
        -----
        1. Collect every unassigned order plus any orders sitting on idle bots.
        2. Build a pairwise distance matrix for all destination coordinates.
        3. Run K-Medoids to split destinations across available bots.
        4. Run Nearest-Neighbour TSP + 2-Opt on each cluster.
        5. Remap coordinate routes back to full order dicts.
        6. Push the ordered route to each bot via set_destinations().
        """
        if not self.unassigned_destinations:
            return

        # Collect idle bots
        available_bots = [bot for bot in self.bots if not bot.delivering_orders]
        print(f"Available robots: {[b.name for b in available_bots]}")

        # Gather all orders not currently being delivered
        non_delivered = list(self.unassigned_destinations)
        for bot in available_bots:
            non_delivered += bot.destinations

        coords = [d["dst"] for d in non_delivered]

        # Build distance matrix (coordinates only — needed for clustering/TSP)
        dist_matrix = self.gen_dist_matrix(WAREHOUSE_LOC, coords)

        # Coordinate → full order dict lookup (covers all sources)
        order_lookup = {d["dst"]: d for d in non_delivered}

        # Cluster coordinates across available bots
        group_labels = self.k_medoids(coords, dist_matrix, len(available_bots))

        # Initialise one empty cluster per bot
        self.clusters = [[] for _ in available_bots]
        for i, label in enumerate(group_labels):
            self.clusters[label].append(coords[i])

        self.unassigned_destinations = []   # All orders are now being processed

        # Plan and assign each cluster's route
        for i, cluster_coords in enumerate(self.clusters):
            route = self.nearest_neighbor_tsp(cluster_coords, WAREHOUSE_LOC, dist_matrix)
            route = self.two_opt(route, WAREHOUSE_LOC, dist_matrix)

            # Convert coordinate list back to order dicts
            route_dicts = [order_lookup[coord] for coord in route]

            self.clusters[i] = route_dicts
            available_bots[i].set_destinations(route_dicts)

        self._update_order_lists()
        self.draw_graph()

    # ── Stepping ─────────────────────────────────────────────────────────────

    def algo_step(self, bot: Robot, state_label: tk.Label,
                  start_btn_handle: tk.Button):
        """
        Advance one robot by a single FSM tick and refresh the UI.

        Also re-enables the "Start Delivery" button when the robot
        returns to AT_WAREHOUSE (i.e. delivering_orders becomes False).
        """
        bot.run_algorithm_fsm()
        state_label.config(text=f"State: {bot.bot_state}")

        if not bot.delivering_orders:
            start_btn_handle.config(state=tk.ACTIVE)

        self.current_path = bot.path

        bot.bagging_results

        self.bagging_var.set(bot.bagging_results)
        self._update_order_lists()
        self.draw_graph()

    # ── Misc graph editing (called from canvas click handlers) ────────────────

    def clear_path(self):
        """Clear the highlighted path on the graph."""
        self.current_path = []
        self.status_var.set("Path cleared.")
        self.draw_graph()


# ──────────────────────────────────────────────────────────────────────────────
# Graph initialisation
# ──────────────────────────────────────────────────────────────────────────────

def init_graph(node_types: dict,
               weights_vertical: list,
               weights_horizontal: list) -> nx.Graph:
    """
    Build the delivery grid as a weighted undirected NetworkX graph.

    Nodes are (row, col) tuples; edges are added for every adjacent
    horizontal and vertical pair.  One edge is removed to create a
    road block near the warehouse.

    Parameters
    ----------
    node_types         : {(r,c): label_string} for all grid nodes
    weights_vertical   : 9×10 list — weights_vertical[r][c] is the cost
                         of the edge (r,c) → (r+1,c)
    weights_horizontal : 10×9 list — weights_horizontal[r][c] is the cost
                         of the edge (r,c) → (r,c+1)
    """
    G = nx.Graph()

    for node, ntype in node_types.items():
        G.add_node(node, kind=ntype[0])

    # Horizontal edges
    for r in range(10):
        for c in range(9):
            G.add_edge((r, c), (r, c + 1), weight=weights_horizontal[r][c])

    # Vertical edges
    for r in range(9):
        for c in range(10):
            G.add_edge((r, c), (r + 1, c), weight=weights_vertical[r][c])

    # Remove one edge to create a forced detour near the warehouse
    G.remove_edge((1, 2), (1, 3))

    return G


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    node_types = {
        # Row 0
        (0,0): "S1", (0,1): "I",  (0,2): "I",  (0,3): "I",  (0,4): "I",
        (0,5): "H1", (0,6): "I",  (0,7): "I",  (0,8): "I",  (0,9): "S2",
        # Row 1
        (1,0): "I",  (1,1): "I",  (1,2): "H2", (1,3): "I",  (1,4): "I",
        (1,5): "I",  (1,6): "I",  (1,7): "I",  (1,8): "H3", (1,9): "I",
        # Row 2
        (2,0): "I",  (2,1): "I",  (2,2): "I",  (2,3): "I",  (2,4): "S3",
        (2,5): "I",  (2,6): "I",  (2,7): "I",  (2,8): "I",  (2,9): "I",
        # Row 3
        (3,0): "I",  (3,1): "H4", (3,2): "I",  (3,3): "I",  (3,4): "I",
        (3,5): "I",  (3,6): "I",  (3,7): "H5", (3,8): "I",  (3,9): "I",
        # Row 4
        (4,0): "I",  (4,1): "I",  (4,2): "I",  (4,3): "I",  (4,4): "I",
        (4,5): "I",  (4,6): "I",  (4,7): "I",  (4,8): "I",  (4,9): "I",
        # Row 5
        (5,0): "S4", (5,1): "I",  (5,2): "I",  (5,3): "I",  (5,4): "I",
        (5,5): "I",  (5,6): "I",  (5,7): "H6", (5,8): "I",  (5,9): "I",
        # Row 6
        (6,0): "I",  (6,1): "I",  (6,2): "H7", (6,3): "I",  (6,4): "I",
        (6,5): "I",  (6,6): "I",  (6,7): "I",  (6,8): "I",  (6,9): "S5",
        # Row 7
        (7,0): "I",  (7,1): "I",  (7,2): "I",  (7,3): "I",  (7,4): "H8",
        (7,5): "I",  (7,6): "I",  (7,7): "I",  (7,8): "I",  (7,9): "I",
        # Row 8
        (8,0): "I",  (8,1): "H9", (8,2): "I",  (8,3): "I",  (8,4): "I",
        (8,5): "I",  (8,6): "I",  (8,7): "I",  (8,8): "I",  (8,9): "I",
        # Row 9
        (9,0): "S6", (9,1): "I",  (9,2): "I",  (9,3): "I",  (9,4): "I",
        (9,5): "H10",(9,6): "I",  (9,7): "I",  (9,8): "I",  (9,9): "S7",
    }

    # Edge costs — horizontal: (r,c)→(r,c+1), 10 rows × 9 edges
    weights_horizontal = [
        [2, 4, 3, 5, 2, 3, 4, 2, 3],   # row 0
        [3, 2, 5, 2, 4, 3, 2, 6, 2],   # row 1
        [4, 3, 2, 4, 3, 2, 5, 3, 2],   # row 2
        [2, 5, 3, 2, 4, 3, 2, 4, 3],   # row 3
        [3, 2, 4, 3, 5, 2, 3, 2, 4],   # row 4
        [4, 3, 2, 5, 3, 2, 4, 3, 2],   # row 5
        [2, 4, 3, 2, 5, 3, 2, 4, 3],   # row 6
        [3, 2, 5, 3, 2, 4, 3, 2, 5],   # row 7
        [4, 3, 2, 4, 3, 5, 2, 3, 2],   # row 8
        [2, 3, 4, 2, 5, 3, 2, 4, 3],   # row 9
    ]

    # Edge costs — vertical: (r,c)→(r+1,c), 9 transitions × 10 edges
    weights_vertical = [
        [3, 2, 4, 3, 2, 5, 3, 2, 4, 3],   # rows 0→1
        [6, 3, 2, 4, 3, 2, 5, 3, 2, 4],   # rows 1→2
        [2, 4, 3, 2, 5, 3, 2, 4, 3, 2],   # rows 2→3
        [4, 2, 5, 3, 2, 4, 3, 2, 5, 3],   # rows 3→4
        [3, 5, 2, 4, 3, 2, 4, 3, 2, 5],   # rows 4→5
        [2, 3, 4, 2, 5, 3, 2, 4, 3, 2],   # rows 5→6
        [4, 2, 3, 5, 2, 4, 3, 2, 4, 3],   # rows 6→7
        [3, 4, 2, 3, 4, 2, 5, 3, 2, 4],   # rows 7→8
        [2, 3, 5, 2, 3, 4, 2, 3, 5, 2],   # rows 8→9
    ]

    G = init_graph(node_types, weights_vertical, weights_horizontal)

    root = tk.Tk()
    app  = FoodieApp(root, G, node_types)
    root.mainloop()


if __name__ == "__main__":
    main()