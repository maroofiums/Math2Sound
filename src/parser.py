"""Parser: turn a string such as "sin(x) + cos(2*x)" into a callable f(x).

Safe by design: we parse with Python's `ast` module and only allow a small
whitelist of nodes, so user input is never passed to eval().
"""
import ast
from typing import Callable

import numpy as np

FUNCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
}
CONSTANTS = {"pi": np.pi, "e": np.e}
BINARY_OPS = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
    ast.Div: np.divide, ast.Pow: np.power,
}
UNARY_OPS = {ast.UAdd: lambda v: v, ast.USub: np.negative}


def _evaluate(node: ast.AST, x: np.ndarray) -> np.ndarray:
    """Recursively evaluate an AST node for the array x."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, x)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return np.full_like(x, float(node.value), dtype=float)
    if isinstance(node, ast.Name):
        if node.id == "x":
            return x
        if node.id in CONSTANTS:
            return np.full_like(x, CONSTANTS[node.id], dtype=float)
        raise ValueError(f"Unknown name: {node.id!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPS:
        left = _evaluate(node.left, x)
        right = _evaluate(node.right, x)
        return BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
        return UNARY_OPS[type(node.op)](_evaluate(node.operand, x))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in FUNCTIONS and len(node.args) == 1
            and not node.keywords):
        return FUNCTIONS[node.func.id](_evaluate(node.args[0], x))
    raise ValueError(f"Unsupported syntax: {ast.dump(node)[:60]}")


def parse_expression(expr: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return f such that f(x_array) -> y_array.  '^' means power."""
    if not expr or not expr.strip():
        raise ValueError("Empty expression")
    tree = ast.parse(expr.replace("^", "**").strip(), mode="eval")

    def f(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):  # 1/0, log(-1) -> inf/nan, handled later
            return _evaluate(tree, x)

    # Validate once on a probe so syntax errors surface at parse time.
    f(np.array([0.5, 1.0]))
    return f
