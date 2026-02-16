from enum import Enum


class NodeType(Enum):
    EVENT = "dynamic_event"
    ROOT = "root"
    SECTION = "section"
    CONTAINER = "container"
    LEAF = "leaf"
