from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any, Optional, Iterable, Iterator, Sequence, Mapping, MutableMapping,
    Hashable, Callable, Protocol, Generic, TypeVar, Tuple, Dict, List, Set,
    Union, overload, Literal, ClassVar, final, runtime_checkable,
)

S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")

Coord = Hashable
Coords = Tuple[Coord, ...]

__all__ = [
    "ABC","abstractmethod","dataclass","Any","Optional","Iterable","Iterator","Sequence",
    "Mapping","MutableMapping","Hashable","Callable","Protocol","Generic","TypeVar","Tuple",
    "Dict","List","Set","Union","overload","Literal","ClassVar","final","runtime_checkable",
    "S","T","U","Coord","Coords",
]
