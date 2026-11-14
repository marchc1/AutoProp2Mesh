"""Reads the entities out of a decoded AdvDupe2 table, whatever their class."""

from .ad2 import Angle, Vector


class DupeEntity:
    """One entity from a dupe. pos/ang are relative to the paste origin."""

    def __init__(self, index, cls, model, pos, ang, parent, table):
        self.index = index
        self.cls = cls
        self.model = model
        self.pos = pos
        self.ang = ang
        self.parent = parent    # dupe index of the parent entity, or None
        self.table = table      # the raw entity table


def _str(v, default=""):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, str):
        return v
    return default


def _index(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def read_entities(dupe):
    """List of DupeEntity, sorted by dupe index."""
    ents = dupe.get("Entities") if isinstance(dupe, dict) else None
    if not isinstance(ents, dict):
        raise ValueError("dupe has no Entities table")
    out = []
    for key, e in ents.items():
        index = _index(key)
        if index is None or not isinstance(e, dict):
            continue
        pos, ang = e.get("Pos"), e.get("Angle")
        phys = e.get("PhysicsObjects")
        if isinstance(phys, dict):
            root = phys.get(0)
            if isinstance(root, dict):
                pos = root.get("Pos", pos)
                ang = root.get("Angle", ang)
        info = e.get("BuildDupeInfo")
        parent = _index(info.get("DupeParentID")) if isinstance(info, dict) else None
        out.append(DupeEntity(
            index, _str(e.get("Class")), _str(e.get("Model")),
            pos if isinstance(pos, Vector) else Vector(),
            ang if isinstance(ang, Angle) else Angle(),
            parent, e))
    out.sort(key=lambda d: d.index)
    return out
