#!/usr/bin/env python3
"""
Maze Solver & Visualizer
=========================

A terminal-based maze pathfinding visualizer built with `curses`.

Features
--------
- Multiple pathfinding algorithms: BFS, DFS, Dijkstra, A*
- Live, animated visualization of the search process in the terminal
- Random, solvable maze generation (recursive backtracker)
- Load mazes from plain-text files, or save generated mazes to disk
- Export the discovered solution path to a JSON/text file
- Clean CLI powered by argparse
- Logging to a file (stdout is reserved for curses)
- Resize-safe rendering with a legend and a live statistics panel

Usage
-----
    python maze_solver.py                          # solve the built-in demo maze with BFS
    python maze_solver.py -a astar                  # solve with A*
    python maze_solver.py -a dijkstra -d 0.02        # faster animation
    python maze_solver.py --no-animation             # skip animation, show result instantly
    python maze_solver.py --generate 15              # generate & solve a random 15x15 maze
    python maze_solver.py --generate 21 --save-maze mymaze.txt
    python maze_solver.py --maze mymaze.txt -a dfs
    python maze_solver.py --maze mymaze.txt -o solution.json

Maze file format
----------------
Plain text, one row per line, using these characters:
    '#'  wall
    ' '  open floor
    'O'  start (exactly one required)
    'X'  goal  (exactly one required)

Author: (your name here)
License: MIT
"""

from __future__ import annotations

import argparse
import curses
import heapq
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

Coordinate = tuple[int, int]

LOG_FILE = "maze_solver.log"
logger = logging.getLogger("maze_solver")


# --------------------------------------------------------------------------- #
# Domain model
# --------------------------------------------------------------------------- #

class Cell(str, Enum):
    """Characters used to represent a maze on disk and in memory."""

    WALL = "#"
    EMPTY = " "
    START = "O"
    GOAL = "X"


DEFAULT_MAZE: list[list[str]] = [
    ["#", "O", "#", "#", "#", "#", "#", "#", "#"],
    ["#", " ", " ", " ", " ", " ", " ", " ", "#"],
    ["#", " ", "#", "#", " ", "#", "#", " ", "#"],
    ["#", " ", "#", " ", " ", " ", "#", " ", "#"],
    ["#", " ", "#", " ", "#", " ", "#", " ", "#"],
    ["#", " ", "#", " ", "#", " ", "#", " ", "#"],
    ["#", " ", "#", " ", "#", " ", "#", "#", "#"],
    ["#", " ", " ", " ", " ", " ", " ", " ", "#"],
    ["#", "#", "#", "#", "#", "#", "#", "X", "#"],
]


class MazeError(Exception):
    """Raised when a maze fails validation (bad shape, missing start/goal, ...)."""


@dataclass
class Maze:
    """An immutable-ish wrapper around a 2D grid of maze cells."""

    grid: list[list[str]]
    start: Coordinate = field(init=False)
    goal: Coordinate = field(init=False)

    def __post_init__(self) -> None:
        self._validate_rectangular()
        self.start = self._find(Cell.START.value)
        self.goal = self._find(Cell.GOAL.value)

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def _validate_rectangular(self) -> None:
        if not self.grid or not self.grid[0]:
            raise MazeError("Maze is empty.")
        width = len(self.grid[0])
        for i, row in enumerate(self.grid):
            if len(row) != width:
                raise MazeError(f"Row {i} has length {len(row)}, expected {width}.")

    def _find(self, target: str) -> Coordinate:
        matches = [
            (r, c)
            for r, row in enumerate(self.grid)
            for c, value in enumerate(row)
            if value == target
        ]
        if len(matches) != 1:
            raise MazeError(
                f"Expected exactly one '{target}' cell, found {len(matches)}."
            )
        return matches[0]

    def is_walkable(self, pos: Coordinate) -> bool:
        r, c = pos
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        return self.grid[r][c] != Cell.WALL.value

    def neighbors(self, pos: Coordinate) -> list[Coordinate]:
        r, c = pos
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [p for p in candidates if self.is_walkable(p)]

    @classmethod
    def from_file(cls, path: str | Path) -> "Maze":
        text = Path(path).read_text(encoding="utf-8").splitlines()
        grid = [list(line) for line in text if line != ""]
        return cls(grid)

    @classmethod
    def default(cls) -> "Maze":
        return cls([row.copy() for row in DEFAULT_MAZE])

    def to_text(self) -> str:
        return "\n".join("".join(row) for row in self.grid)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_text(), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Maze generation
# --------------------------------------------------------------------------- #

def generate_maze(size: int, rng: Optional[random.Random] = None) -> Maze:
    """Generate a random, guaranteed-solvable maze using a recursive backtracker.

    `size` is the number of "cells" per side; the resulting grid will be
    (2*size + 1) characters wide/tall to leave room for walls between cells.
    """
    if size < 2:
        raise ValueError("size must be >= 2")

    rng = rng or random.Random()
    width, height = size, size
    grid_w, grid_h = width * 2 + 1, height * 2 + 1
    grid = [[Cell.WALL.value for _ in range(grid_w)] for _ in range(grid_h)]

    def carve(cell: Coordinate, visited: set[Coordinate]) -> None:
        cx, cy = cell
        grid[cy * 2 + 1][cx * 2 + 1] = Cell.EMPTY.value
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        rng.shuffle(directions)
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                visited.add((nx, ny))
                grid[cy * 2 + 1 + dy][cx * 2 + 1 + dx] = Cell.EMPTY.value
                carve((nx, ny), visited)

    start_cell = (0, 0)
    carve(start_cell, {start_cell})

    grid[1][1] = Cell.START.value
    grid[grid_h - 2][grid_w - 2] = Cell.GOAL.value
    return Maze(grid)


# --------------------------------------------------------------------------- #
# Search algorithms
# --------------------------------------------------------------------------- #

@dataclass
class SearchResult:
    """Outcome of running a search algorithm against a maze."""

    algorithm: str
    path: Optional[list[Coordinate]]
    explored_order: list[Coordinate]
    elapsed_seconds: float

    @property
    def solved(self) -> bool:
        return self.path is not None

    @property
    def path_length(self) -> int:
        return len(self.path) if self.path else 0

    @property
    def nodes_explored(self) -> int:
        return len(self.explored_order)


def _reconstruct_path(came_from: dict[Coordinate, Coordinate], end: Coordinate,
                       start: Coordinate) -> list[Coordinate]:
    path = [end]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def solve_bfs(maze: Maze) -> SearchResult:
    from collections import deque

    frontier: deque[Coordinate] = deque([maze.start])
    came_from: dict[Coordinate, Coordinate] = {}
    visited = {maze.start}
    explored_order: list[Coordinate] = []

    t0 = time.perf_counter()
    while frontier:
        current = frontier.popleft()
        explored_order.append(current)
        if current == maze.goal:
            path = _reconstruct_path(came_from, current, maze.start)
            return SearchResult("BFS", path, explored_order, time.perf_counter() - t0)

        for neighbor in maze.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                frontier.append(neighbor)

    return SearchResult("BFS", None, explored_order, time.perf_counter() - t0)


def solve_dfs(maze: Maze) -> SearchResult:
    stack: list[Coordinate] = [maze.start]
    came_from: dict[Coordinate, Coordinate] = {}
    visited = {maze.start}
    explored_order: list[Coordinate] = []

    t0 = time.perf_counter()
    while stack:
        current = stack.pop()
        explored_order.append(current)
        if current == maze.goal:
            path = _reconstruct_path(came_from, current, maze.start)
            return SearchResult("DFS", path, explored_order, time.perf_counter() - t0)

        for neighbor in maze.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)

    return SearchResult("DFS", None, explored_order, time.perf_counter() - t0)


def solve_dijkstra(maze: Maze) -> SearchResult:
    # All moves cost 1, so this behaves like BFS but demonstrates the
    # classic priority-queue formulation (useful if weights are added later).
    counter = 0
    frontier: list[tuple[int, int, Coordinate]] = [(0, counter, maze.start)]
    came_from: dict[Coordinate, Coordinate] = {}
    cost_so_far = {maze.start: 0}
    explored_order: list[Coordinate] = []

    t0 = time.perf_counter()
    while frontier:
        cost, _, current = heapq.heappop(frontier)
        explored_order.append(current)
        if current == maze.goal:
            path = _reconstruct_path(came_from, current, maze.start)
            return SearchResult("Dijkstra", path, explored_order, time.perf_counter() - t0)

        for neighbor in maze.neighbors(current):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(frontier, (new_cost, counter, neighbor))

    return SearchResult("Dijkstra", None, explored_order, time.perf_counter() - t0)


def _manhattan(a: Coordinate, b: Coordinate) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def solve_astar(maze: Maze) -> SearchResult:
    counter = 0
    frontier: list[tuple[int, int, Coordinate]] = [(0, counter, maze.start)]
    came_from: dict[Coordinate, Coordinate] = {}
    cost_so_far = {maze.start: 0}
    explored_order: list[Coordinate] = []

    t0 = time.perf_counter()
    while frontier:
        _, _, current = heapq.heappop(frontier)
        explored_order.append(current)
        if current == maze.goal:
            path = _reconstruct_path(came_from, current, maze.start)
            return SearchResult("A*", path, explored_order, time.perf_counter() - t0)

        for neighbor in maze.neighbors(current):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + _manhattan(neighbor, maze.goal)
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(frontier, (priority, counter, neighbor))

    return SearchResult("A*", None, explored_order, time.perf_counter() - t0)


ALGORITHMS: dict[str, Callable[[Maze], SearchResult]] = {
    "bfs": solve_bfs,
    "dfs": solve_dfs,
    "dijkstra": solve_dijkstra,
    "astar": solve_astar,
}


# --------------------------------------------------------------------------- #
# Visualization (curses)
# --------------------------------------------------------------------------- #

class Visualizer:
    """Renders a maze and an animated search in the terminal via curses."""

    COLOR_WALL = 1
    COLOR_EXPLORED = 2
    COLOR_PATH = 3
    COLOR_ENDPOINTS = 4
    COLOR_TEXT = 5

    def __init__(self, stdscr: "curses._CursesWindow", maze: Maze, delay: float) -> None:
        self.stdscr = stdscr
        self.maze = maze
        self.delay = delay
        self._setup_colors()
        curses.curs_set(0)
        self.stdscr.nodelay(True)

    def _setup_colors(self) -> None:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(self.COLOR_WALL, curses.COLOR_WHITE, -1)
        curses.init_pair(self.COLOR_EXPLORED, curses.COLOR_BLUE, -1)
        curses.init_pair(self.COLOR_PATH, curses.COLOR_RED, -1)
        curses.init_pair(self.COLOR_ENDPOINTS, curses.COLOR_GREEN, -1)
        curses.init_pair(self.COLOR_TEXT, curses.COLOR_YELLOW, -1)

    def _check_terminal_size(self) -> bool:
        max_y, max_x = self.stdscr.getmaxyx()
        needed_y = self.maze.rows + 6
        needed_x = self.maze.cols * 2 + 2
        return max_y >= needed_y and max_x >= needed_x

    def _draw_maze(self, explored: set[Coordinate], path: set[Coordinate]) -> None:
        for r, row in enumerate(self.maze.grid):
            for c, value in enumerate(row):
                if (r, c) in path:
                    ch, color = "*", self.COLOR_PATH
                elif (r, c) in (self.maze.start, self.maze.goal):
                    ch, color = value, self.COLOR_ENDPOINTS
                elif (r, c) in explored:
                    ch, color = ".", self.COLOR_EXPLORED
                elif value == Cell.WALL.value:
                    ch, color = value, self.COLOR_WALL
                else:
                    ch, color = value, 0
                try:
                    self.stdscr.addstr(r + 2, c * 2, ch, curses.color_pair(color))
                except curses.error:
                    pass  # Terminal too small for this cell; ignore gracefully.

    def _draw_header(self, algorithm: str) -> None:
        title = f" Maze Solver — algorithm: {algorithm.upper()} "
        self.stdscr.addstr(0, 0, title, curses.color_pair(self.COLOR_TEXT) | curses.A_BOLD)

    def _draw_footer(self, result: Optional[SearchResult], step: int, total: int) -> None:
        base_row = self.maze.rows + 2
        legend = "Legend:  O/X endpoints   . explored   * path   # wall"
        self.stdscr.addstr(base_row, 0, legend, curses.color_pair(self.COLOR_TEXT))

        if result is None:
            status = f"Exploring... step {step}/{total}"
        elif result.solved:
            status = (
                f"Solved! path length={result.path_length}  "
                f"nodes explored={result.nodes_explored}  "
                f"time={result.elapsed_seconds * 1000:.1f}ms"
            )
        else:
            status = (
                f"No path found. nodes explored={result.nodes_explored}  "
                f"time={result.elapsed_seconds * 1000:.1f}ms"
            )
        self.stdscr.addstr(base_row + 1, 0, status, curses.color_pair(self.COLOR_TEXT))
        self.stdscr.addstr(base_row + 2, 0, "Press any key to exit.",
                            curses.color_pair(self.COLOR_TEXT))

    def animate(self, result: SearchResult, animate_steps: bool) -> None:
        if not self._check_terminal_size():
            raise RuntimeError(
                "Terminal window is too small to display this maze. "
                "Please resize your terminal and try again."
            )

        explored: set[Coordinate] = set()
        total = len(result.explored_order)

        if animate_steps:
            for step, node in enumerate(result.explored_order, start=1):
                explored.add(node)
                self.stdscr.erase()
                self._draw_header(result.algorithm)
                self._draw_maze(explored, set())
                self._draw_footer(None, step, total)
                self.stdscr.refresh()
                if self.stdscr.getch() != -1:
                    break  # allow early exit on keypress
                time.sleep(self.delay)
        else:
            explored.update(result.explored_order)

        path_cells = set(result.path) if result.path else set()
        self.stdscr.erase()
        self._draw_header(result.algorithm)
        self._draw_maze(explored, path_cells)
        self._draw_footer(result, total, total)
        self.stdscr.refresh()

        self.stdscr.nodelay(False)
        self.stdscr.getch()


# --------------------------------------------------------------------------- #
# CLI plumbing
# --------------------------------------------------------------------------- #

def configure_logging(level: str) -> None:
    logging.basicConfig(
        filename=LOG_FILE,
        filemode="a",
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maze_solver",
        description="Solve and visualize mazes with BFS, DFS, Dijkstra, or A*.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--maze", type=str, default=None,
        help="Path to a maze text file. Uses a built-in demo maze if omitted.",
    )
    parser.add_argument(
        "-a", "--algorithm", type=str, choices=sorted(ALGORITHMS.keys()), default="bfs",
        help="Search algorithm to use.",
    )
    parser.add_argument(
        "-d", "--delay", type=float, default=0.05,
        help="Delay in seconds between animation frames.",
    )
    parser.add_argument(
        "--no-animation", action="store_true",
        help="Skip step-by-step animation; render only the final result.",
    )
    parser.add_argument(
        "--generate", type=int, metavar="SIZE", default=None,
        help="Generate a random solvable maze of the given size instead of loading one.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for maze generation (for reproducibility).",
    )
    parser.add_argument(
        "--save-maze", type=str, default=None, metavar="PATH",
        help="Save the (generated or loaded) maze to a text file.",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None, metavar="PATH",
        help="Save the solution (path, stats) to a JSON file.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (written to maze_solver.log).",
    )
    return parser


def load_or_build_maze(args: argparse.Namespace) -> Maze:
    if args.generate is not None:
        rng = random.Random(args.seed) if args.seed is not None else None
        logger.info("Generating random maze of size %s (seed=%s)", args.generate, args.seed)
        maze = generate_maze(args.generate, rng)
    elif args.maze:
        logger.info("Loading maze from %s", args.maze)
        maze = Maze.from_file(args.maze)
    else:
        logger.info("Using built-in demo maze")
        maze = Maze.default()

    if args.save_maze:
        maze.save(args.save_maze)
        logger.info("Maze saved to %s", args.save_maze)

    return maze


def write_solution_output(path: str, result: SearchResult) -> None:
    payload = {
        "algorithm": result.algorithm,
        "solved": result.solved,
        "path_length": result.path_length,
        "nodes_explored": result.nodes_explored,
        "elapsed_seconds": result.elapsed_seconds,
        "path": result.path,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Solution written to %s", path)


def run_curses_app(stdscr: "curses._CursesWindow", maze: Maze, result: SearchResult,
                    delay: float, animate_steps: bool) -> None:
    viz = Visualizer(stdscr, maze, delay)
    viz.animate(result, animate_steps)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    try:
        maze = load_or_build_maze(args)
    except (MazeError, OSError, ValueError) as exc:
        print(f"Error preparing maze: {exc}", file=sys.stderr)
        logger.error("Failed to prepare maze: %s", exc)
        return 1

    solver = ALGORITHMS[args.algorithm]
    logger.info("Solving maze with %s", args.algorithm)
    result = solver(maze)

    if args.output:
        write_solution_output(args.output, result)

    try:
        curses.wrapper(
            run_curses_app, maze, result, args.delay, not args.no_animation
        )
    except RuntimeError as exc:
        print(f"Display error: {exc}", file=sys.stderr)
        logger.error("Display error: %s", exc)
        return 1
    except curses.error as exc:
        print(f"Terminal rendering error: {exc}. Try a larger terminal window.",
              file=sys.stderr)
        logger.error("curses error: %s", exc)
        return 1

    if not result.solved:
        print("No path exists between the start and the goal.")
        return 2

    print(
        f"Solved with {result.algorithm}: "
        f"path length={result.path_length}, "
        f"nodes explored={result.nodes_explored}, "
        f"time={result.elapsed_seconds * 1000:.1f}ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())