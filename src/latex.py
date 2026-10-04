"""Typeset an expression as LaTeX (for the Streamlit math preview).

Uses the same ast tree as the parser, so what you see is what is evaluated.
"""
import ast

_FUNC = {"sin": r"\sin", "cos": r"\cos", "tan": r"\tan", "exp": r"\exp", "log": r"\ln"}


def _num(v) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _is_addsub(n) -> bool:
    return isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub))


def _paren(s: str) -> str:
    return rf"\left({s}\right)"


def _tex(n: ast.AST) -> str:
    if isinstance(n, ast.Constant):
        return _num(n.value)
    if isinstance(n, ast.Name):
        return r"\pi" if n.id == "pi" else n.id
    if isinstance(n, ast.UnaryOp):
        inner = _tex(n.operand)
        if isinstance(n.operand, (ast.UnaryOp,)) or _is_addsub(n.operand):
            inner = _paren(inner)
        return ("-" if isinstance(n.op, ast.USub) else "") + inner
    if isinstance(n, ast.BinOp):
        l, r = n.left, n.right
        if isinstance(n.op, ast.Add):
            rs = _tex(r)
            return f"{_tex(l)}+" + (_paren(rs) if isinstance(r, ast.UnaryOp) else rs)
        if isinstance(n.op, ast.Sub):
            rs = _tex(r)
            return f"{_tex(l)}-" + (_paren(rs) if _is_addsub(r) or isinstance(r, ast.UnaryOp) else rs)
        if isinstance(n.op, ast.Mult):
            ls, rs = _tex(l), _tex(r)
            if _is_addsub(l) or isinstance(l, ast.UnaryOp):
                ls = _paren(ls)
            if _is_addsub(r) or isinstance(r, ast.UnaryOp):
                rs = _paren(rs)
            lead = isinstance(l, ast.Constant) or (isinstance(l, ast.Name) and l.id == "pi") \
                or (isinstance(l, ast.BinOp) and isinstance(l.op, ast.Mult))
            if lead and not isinstance(r, ast.Constant):
                return f"{ls} {rs}"                    # 2x, 2\pi x (space ends the \pi command)
            return rf"{ls}\cdot {rs}"
        if isinstance(n.op, ast.Div):
            return rf"\frac{{{_tex(l)}}}{{{_tex(r)}}}"
        if isinstance(n.op, ast.Pow):
            base = _tex(l)
            if isinstance(l, (ast.BinOp, ast.UnaryOp)):
                base = _paren(base)
            return rf"{base}^{{{_tex(r)}}}"
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and len(n.args) == 1:
        arg = _tex(n.args[0])
        if n.func.id == "sqrt":
            return rf"\sqrt{{{arg}}}"
        if n.func.id == "abs":
            return rf"\left|{arg}\right|"
        return rf"{_FUNC.get(n.func.id, n.func.id)}\left({arg}\right)"
    raise ValueError("Cannot typeset expression")


def to_latex(expr: str) -> str:
    tree = ast.parse(expr.replace("^", "**").strip(), mode="eval")
    return _tex(tree.body)
