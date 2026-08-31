"""Extract ``Class.method -> source`` from a whole Java file (05).

The minimal-pair miner needs the same test method before and after a fix
commit, which means locating it inside a full source file rather than a
FlakeBench fragment.  Annotations come along with the method: tree-sitter puts
them inside the declaration's ``modifiers``, so the node text already reads
``@Test public void foo() {...}``, matching the shape of FlakeBench rows.
"""
from __future__ import annotations

from .structural import get_parser

TYPE_NODES = ("class_declaration", "interface_declaration", "enum_declaration",
              "record_declaration")


def extract_methods(source: str) -> dict[str, str]:
    """``{"Outer.method": text, "Outer.Inner.method": text, "method": text}``.

    The bare method name is included as a key only when it is unambiguous in
    the file, so a caller can fall back to it safely.
    """
    src = source.encode("utf-8")
    tree = get_parser().parse(src)

    out: dict[str, str] = {}
    bare: dict[str, list[str]] = {}

    def text(n) -> str:
        return src[n.start_byte:n.end_byte].decode("utf-8", "replace")

    def walk(node, scope: list[str]) -> None:
        for child in node.named_children:
            if child.type in TYPE_NODES:
                name_node = child.child_by_field_name("name")
                name = text(name_node) if name_node is not None else "?"
                body = child.child_by_field_name("body")
                if body is not None:
                    walk(body, scope + [name])
            elif child.type == "method_declaration":
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    continue
                m = text(name_node)
                body = text(child)
                if scope:
                    out[f"{scope[-1]}.{m}"] = body
                    out[".".join(scope + [m])] = body
                bare.setdefault(m, []).append(body)
            else:
                walk(child, scope)

    walk(tree.root_node, [])
    for m, bodies in bare.items():
        if len(bodies) == 1 and m not in out:
            out[m] = bodies[0]
    return out


def is_test_path(path: str) -> bool:
    p = path.replace("\\", "/")
    if not p.endswith(".java"):
        return False
    lower = p.lower()
    if "/test/" in lower or "/tests/" in lower or "/androidtest/" in lower:
        return True
    base = p.rsplit("/", 1)[-1][:-5]
    return base.endswith(("Test", "Tests", "TestCase", "IT", "ITCase")) or base.startswith("Test")
