"""Valve KeyValues / VDF text parser (VMTs, VDF manifests, mount configs)."""


class KVNode:
    """An ordered multimap. Values are str or KVNode. Lookups ignore case."""
    __slots__ = ("items",)

    def __init__(self):
        self.items = []

    def get(self, key, default=None):
        key = key.lower()
        for k, v in self.items:
            if k.lower() == key:
                return v
        return default

    def get_node(self, key):
        v = self.get(key)
        return v if isinstance(v, KVNode) else None

    def __iter__(self):
        return iter(self.items)

    def to_dict(self):
        """Flattened, lower-cased dict of string values (last one wins)."""
        return {k.lower(): v for k, v in self.items if isinstance(v, str)}


def _tokenize(text):
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n﻿":
            i += 1
        elif c == "/" and text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
        elif c in "{}":
            yield c, False
            i += 1
        elif c == '"':
            j = i + 1
            buf = []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n and text[j + 1] in '"\\':
                    j += 1
                buf.append(text[j])
                j += 1
            yield "".join(buf), True
            i = j + 1
        elif c == "[":
            j = text.find("]", i)
            j = n if j < 0 else j + 1
            # Platform conditionals such as [$WIN32] are ignored; anything
            # else in brackets (e.g. an unquoted "[1 1 1]") is a value.
            if not (text.startswith("[$", i) or text.startswith("[!$", i)):
                yield text[i:j], True
            i = j
        else:
            j = i
            while j < n and text[j] not in ' \t\r\n{}"':
                j += 1
            yield text[i:j], True
            i = j


def parse(text: str) -> KVNode:
    root = KVNode()
    stack = [root]
    key = None
    for tok, is_str in _tokenize(text):
        if not is_str:
            if tok == "{":
                node = KVNode()
                stack[-1].items.append((key if key is not None else "", node))
                stack.append(node)
                key = None
            else:
                if len(stack) > 1:
                    stack.pop()
                key = None
        elif key is None:
            key = tok
        else:
            stack[-1].items.append((key, tok))
            key = None
    return root


def parse_file(path: str) -> KVNode:
    with open(path, "rb") as f:
        return parse(f.read().decode("utf-8", "replace"))
