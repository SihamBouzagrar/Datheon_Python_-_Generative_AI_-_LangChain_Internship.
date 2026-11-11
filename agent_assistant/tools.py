"""Tools for the autonomous AI agent (Task 3):
web search, secure calculator, and local file reader.
"""

import ast
import math
import operator
import time
from datetime import date
from pathlib import Path

from ddgs import DDGS
from langchain_core.tools import tool


# ---------- Configuration ----------

WORKSPACE = Path(__file__).parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)  # avoid confusing "not found" errors

MAX_CHARS = 4000                 # max characters returned by the file reader
MAX_FILE_BYTES = 5_000_000       # refuse to read files larger than 5 MB
MAX_LISTED_FILES = 200           # cap directory listings
MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 1000
MAX_ABS_VALUE = 1e100

SEARCH_RETRIES = 3


# ---------- Tool 1: Web Search ----------

@tool
def web_search(query: str, recent: bool = False) -> str:
    """Search the web using DuckDuckGo for up-to-date information,
    news, facts, prices, and recent events.

    Input:
      query: a short, specific search query.
      recent: set to True for anything that may have changed in the last
        year (current job holders, news, prices, latest versions,
        recent events). Results are then limited to the past year.

    The output starts with today's date so you can judge how old the
    sources are. If sources look outdated or conflict, say so.
    """

    last_error = None

    for attempt in range(SEARCH_RETRIES):
        try:
            kwargs = {"max_results": 5}

            if recent:
                kwargs["timelimit"] = "y"  # d=day, w=week, m=month, y=year

            results = DDGS().text(query, **kwargs)

            if not results:
                return (
                    f"(Today's date: {date.today().isoformat()})\n"
                    "No search results found."
                )

            body = "\n\n".join(
                f"Title: {r.get('title', 'N/A')}\n"
                f"Description: {r.get('body', 'N/A')}\n"
                f"URL: {r.get('href', 'N/A')}"
                for r in results
            )

            return f"(Today's date: {date.today().isoformat()})\n\n{body}"

        except Exception as exc:
            # DuckDuckGo often rate-limits; wait a bit and retry.
            last_error = exc
            if attempt < SEARCH_RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))

    return f"Web search error after {SEARCH_RETRIES} attempts: {last_error}"


# ---------- Tool 2: Secure Python Calculator ----------

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _check_result(result):
    """Reject complex, non-finite, or excessively large results."""

    if isinstance(result, complex):
        raise ValueError("Complex results are not supported.")

    if isinstance(result, (int, float)) and (
        not math.isfinite(result) or abs(result) > MAX_ABS_VALUE
    ):
        raise ValueError("Result too large.")

    return result


def _eval(node):
    """Safely evaluate an AST mathematical expression."""

    # Numeric constants
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        if abs(node.value) > MAX_ABS_VALUE:
            raise ValueError("Number too large.")
        return node.value

    # Binary operations
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left = _eval(node.left)
        right = _eval(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError("Exponent too large.")

        return _check_result(_OPERATORS[type(node.op)](left, right))

    # Unary operations
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _check_result(_OPERATORS[type(node.op)](_eval(node.operand)))

    # Mathematical constants
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]

    # Mathematical functions
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        if len(node.args) > 10:
            raise ValueError("Too many function arguments.")

        args = [_eval(arg) for arg in node.args]
        return _check_result(_FUNCTIONS[node.func.id](*args))

    raise ValueError("Unsupported mathematical expression.")


@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions safely.

    Supports:
    +, -, *, /, **, %, //, parentheses,
    sqrt, log, log10, exp, sin, cos, tan, floor, ceil,
    abs, round, min, max, pi and e.

    For percentages, write them as multiplication, e.g. 15% of 2340
    is 0.15 * 2340 (the % symbol means modulo).

    Example: (1200 * 1.2) / 3 + sqrt(16)
    Use this tool for all mathematical calculations.
    """

    try:
        if len(expression) > MAX_EXPRESSION_LENGTH:
            return "Error: expression too long."

        tree = ast.parse(expression.strip(), mode="eval")
        return str(_eval(tree.body))

    except ZeroDivisionError:
        return "Error: division by zero."

    except (ValueError, TypeError, OverflowError) as exc:
        return f"Error: {exc}"

    except Exception as exc:
        return f"Error evaluating expression: {exc}"


# ---------- Tool 3: Local File Reader ----------

ALLOWED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".py", ".pdf"}


@tool
def read_local_file(path: str) -> str:
    """Read a local text file or PDF from the workspace folder.

    Supported extensions:
    .txt, .md, .csv, .json, .py, .pdf

    Input examples:
    'budget.txt'
    'notes.md'
    '.'

    Use '.' to list available files.
    """

    try:
        workspace = WORKSPACE.resolve()
        target = (workspace / path).resolve()

        # Security: prevent access outside workspace
        if not target.is_relative_to(workspace):
            return "Error: access denied (outside workspace)."

        # List available files
        if target.is_dir():
            files = sorted(
                str(p.relative_to(workspace))
                for p in target.rglob("*")
                if p.is_file()
            )

            if not files:
                return "Workspace is empty."

            shown = files[:MAX_LISTED_FILES]
            extra = len(files) - len(shown)
            listing = "Available files:\n" + "\n".join(shown)

            if extra > 0:
                listing += f"\n...and {extra} more files."

            return listing

        # Check file existence
        if not target.exists():
            return (
                f"Error: file '{path}' not found. "
                "Use '.' to list available files."
            )

        # Restrict supported file types
        suffix = target.suffix.lower()

        if suffix not in ALLOWED_EXTENSIONS:
            return "Error: unsupported file format."

        # Refuse very large files before reading them
        if target.stat().st_size > MAX_FILE_BYTES:
            return "Error: file too large (limit is 5 MB)."

        # Read PDF
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(target))

            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )

        # Read text files
        else:
            text = target.read_text(encoding="utf-8", errors="replace")

        # Limit output size
        if len(text) > MAX_CHARS:
            text = (
                text[:MAX_CHARS]
                + f"\n...[truncated, {len(text)} characters total]"
            )

        return text

    except Exception as exc:
        return f"Error reading file: {exc}"


# ---------- Available Agent Tools ----------

TOOLS = [
    web_search,
    calculator,
    read_local_file,
]


# ---------- Manual Testing ----------

if __name__ == "__main__":

    print("\n===== CALCULATOR TEST =====")
    print(calculator.invoke({"expression": "(1200 * 1.2) / 3 + sqrt(16)"}))
    print(calculator.invoke({"expression": "0.15 * 2340"}))

    print("\n===== CALCULATOR SECURITY TEST =====")
    print(calculator.invoke({"expression": "__import__('os').system('dir')"}))
    print(calculator.invoke({"expression": "9 ** 9 ** 9"}))
    print(calculator.invoke({"expression": "(-8) ** 0.5"}))

    print("\n===== FILE READER TEST =====")
    print(read_local_file.invoke({"path": "."}))

    print("\n===== FILE SECURITY TEST =====")
    print(read_local_file.invoke({"path": "../../.env"}))

    print("\n===== WEB SEARCH TEST (recent) =====")
    print(
        web_search.invoke(
            {"query": "Netflix co-CEO chairman", "recent": True}
        )[:600]
    )