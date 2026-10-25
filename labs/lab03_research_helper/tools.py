"""Tools for the research helper. Each docstring is what the model reads to
decide when to call the tool, so it's written for the model."""

import ast
import html
import operator
import re
from datetime import date
from urllib.parse import quote

import httpx
from langchain_core.tools import tool

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
# Wikipedia's API policy asks for a User-Agent that says who you are.
HEADERS = {"User-Agent": "langgraph-lab/0.1 (learning project; https://github.com/jinxx45/langgraph-lab)"}
MAX_OUTPUT_CHARS = 2000  # keep tool results small: they all go back into the prompt


def _http_get(url: str, params: dict | None = None) -> dict:
    resp = httpx.get(url, params=params, headers=HEADERS, timeout=10, follow_redirects=True)
    resp.raise_for_status()
    return resp.json()


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return up to 5 matching article titles with a short snippet.
    Use this first to find the exact title of the article you need."""
    data = _http_get(
        WIKI_API,
        {"action": "query", "list": "search", "srsearch": query, "srlimit": 5, "format": "json"},
    )
    hits = data["query"]["search"]
    if not hits:
        return "No results."
    # Snippets come as HTML: <span> highlight tags and entities like &amp;.
    return "\n".join(f"- {h['title']}: {html.unescape(re.sub(r'<[^>]+>', '', h['snippet']))}" for h in hits)


@tool
def read_wikipedia(title: str) -> str:
    """Read the summary of a Wikipedia article. Pass the exact title from search_wikipedia."""
    data = _http_get(WIKI_SUMMARY + quote(title.replace(" ", "_")))
    return (data.get("extract") or "No summary available.")[:MAX_OUTPUT_CHARS]


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node: ast.AST) -> float:
    # Walk the parsed expression and allow only numbers and arithmetic.
    # Never eval() model output: "__import__('os').system(...)" is valid Python.
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("only numbers and + - * / % ** ( ) are allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression like '(2 + 3) * 4' or '2026 - 1653'.
    Only numbers, + - * / % ** and parentheses. Use this for ANY arithmetic."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as err:
        raise ValueError(f"not a valid expression: {expression!r}") from err
    return str(_evaluate(tree.body))


@tool
def today() -> str:
    """Today's date as YYYY-MM-DD. Use it for questions about 'now', 'how long ago' or ages."""
    return date.today().isoformat()


TOOLS = [search_wikipedia, read_wikipedia, calculator, today]

# Failures we expect and want the model to recover from. Anything else is a
# bug and should crash loudly instead of being hidden from us.
EXPECTED_TOOL_ERRORS = (ValueError, ArithmeticError, httpx.HTTPError)
