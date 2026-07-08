Maze Solver & Visualizer

A terminal-based maze pathfinding visualizer written in pure Python (standard library only). It generates or loads mazes, solves them with four classic search algorithms, and animates the search live in your terminal with curses.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
  <img alt="Dependencies" src="https://img.shields.io/badge/dependencies-none-brightgreen.svg">
</p>
Features


Four algorithms: BFS, DFS, Dijkstra, A*
Live animated visualization of the search process
Random, guaranteed-solvable maze generation (recursive backtracker, seedable)
Load/save mazes as plain text files
Export solution + stats to JSON
Logging to file, resize-safe rendering, graceful error handling


Installation
Requires Python 3.10+, no dependencies.
python maze.py

Project Structure

Single-file module with maze logic, search algorithms, and rendering kept independent — the solve_* functions are pure (no I/O, no curses) and easy to unit test.


