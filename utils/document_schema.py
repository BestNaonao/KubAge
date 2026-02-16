from enum import Enum


class NodeType(str, Enum):
    EVENT = "dynamic_event"
    ROOT = "root"
    SECTION = "section"
    CONTAINER = "container"
    LEAF = "leaf"

class SourceType(str, Enum):
    DYNAMIC = "dynamic_event"
    ANCHOR = "anchor"
    PARENT = "parent"
    SIBLING = "sibling"
    LINK = "link"
    UNKNOWN = "unknown"
