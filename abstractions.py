"""
abstractions.py
===============
AIRIS-ISL Core Module — Phase 1: Abstraction Loading

An Abstraction is the smallest unit of knowledge the system holds
about any entity in its environment. It is NOT a text description —
it is a structured, measurable, verifiable unit of experience.

Author: A.M. Almurish
Project: AIRIS-ISL
Version: 1.0.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import json
import copy


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
MIN_CONFIDENCE        = 0.30   # below this → abstraction is quarantined
CONFIDENCE_DECAY      = 0.15   # per generation
VERIFY_BOOST          = 0.05   # confidence gain on successful verification
VERIFY_PENALTY        = 0.10   # confidence loss on failed verification
MAX_GENERATION        = 5      # maximum derivation depth


# ─────────────────────────────────────────────
#  Abstraction Dataclass
# ─────────────────────────────────────────────
@dataclass
class Abstraction:
    """
    A single unit of knowledge about one entity in the environment.

    Fields:
        entity_id   : int   — matches the grid value in AIRIS (0, 2, 3...)
        name        : str   — human-readable name
        properties  : dict  — measurable characteristics (passable, dangerous...)
        behavior    : list  — known interaction rules as strings
        confidence  : float — reliability score [0.0 → 1.0]
        generation  : int   — 0=seed, 1=derived, 2=derived²...
        source      : str   — "seed" | "dream" | "reality"
        verified    : bool  — confirmed by at least one real interaction
    """
    entity_id  : int
    name       : str
    properties : dict[str, Any]      = field(default_factory=dict)
    behavior   : list[str]           = field(default_factory=list)
    confidence : float               = 1.0
    generation : int                 = 0
    source     : str                 = "seed"
    verified   : bool                = True

    # ── post-init validation ──────────────────
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        if self.generation < 0:
            raise ValueError(f"generation must be >= 0, got {self.generation}")
        if self.source not in ("seed", "dream", "reality"):
            raise ValueError(f"source must be seed|dream|reality, got {self.source}")

    # ── computed property ─────────────────────
    @property
    def is_active(self) -> bool:
        """Abstraction is active only if confidence is above minimum."""
        return self.confidence >= MIN_CONFIDENCE

    @property
    def is_passable(self) -> bool:
        return bool(self.properties.get("passable", False))

    @property
    def is_dangerous(self) -> bool:
        return bool(self.properties.get("dangerous", False))

    @property
    def is_collectable(self) -> bool:
        return bool(self.properties.get("collectable", False))

    @property
    def is_goal(self) -> bool:
        return bool(self.properties.get("goal", False))

    # ── confidence updates ────────────────────
    def verify_success(self) -> None:
        """Called when reality confirms this abstraction's prediction."""
        self.confidence = min(1.0, self.confidence + VERIFY_BOOST)
        self.verified = True

    def verify_failure(self) -> None:
        """Called when reality contradicts this abstraction's prediction."""
        self.confidence = max(0.0, self.confidence - VERIFY_PENALTY)
        # NOTE: we NEVER delete — Rule 3: Correct, Never Delete

    def correct(self, new_properties: dict, new_behavior: list | None = None) -> None:
        """
        Correct this abstraction based on observed reality.
        Confidence is penalised, but knowledge is preserved and updated.
        """
        self.properties.update(new_properties)
        if new_behavior:
            for b in new_behavior:
                if b not in self.behavior:
                    self.behavior.append(b)
        self.verify_failure()

    # ── derivation ────────────────────────────
    def derive(self, new_properties: dict, source: str = "dream") -> "Abstraction":
        """
        Create a child abstraction (one generation deeper).
        Confidence decays by CONFIDENCE_DECAY per generation.
        """
        next_gen = min(self.generation + 1, MAX_GENERATION)
        new_conf = max(MIN_CONFIDENCE, 1.0 - (CONFIDENCE_DECAY * next_gen))
        child = copy.deepcopy(self)
        child.properties.update(new_properties)
        child.confidence  = new_conf
        child.generation  = next_gen
        child.source      = source
        child.verified    = False
        return child

    # ── serialization ─────────────────────────
    def to_dict(self) -> dict:
        return {
            "entity_id"  : self.entity_id,
            "name"       : self.name,
            "properties" : self.properties,
            "behavior"   : self.behavior,
            "confidence" : round(self.confidence, 4),
            "generation" : self.generation,
            "source"     : self.source,
            "verified"   : self.verified,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Abstraction":
        return cls(**data)

    def __repr__(self) -> str:
        status = "✅" if self.is_active else "⚠️ QUARANTINED"
        return (
            f"Abstraction({self.name!r} | "
            f"conf={self.confidence:.2f} | "
            f"gen={self.generation} | "
            f"src={self.source} | {status})"
        )


# ─────────────────────────────────────────────
#  Abstraction Store
# ─────────────────────────────────────────────
class AbstractionStore:
    """
    The central registry of all Abstractions the agent knows.

    Provides:
        get(entity_id)     → Abstraction or None
        get_active(id)     → only if confidence >= MIN_CONFIDENCE
        register(abs)      → add or update
        load_seeds()       → populate with SEED_ABSTRACTIONS
        active_all()       → all abstractions above threshold
        quarantined_all()  → all below threshold (for inspection)
        snapshot()         → full serializable dict
    """

    def __init__(self):
        self._store: dict[int, Abstraction] = {}

    # ── CRUD ──────────────────────────────────
    def register(self, abstraction: Abstraction) -> None:
        """Add or overwrite an abstraction."""
        self._store[abstraction.entity_id] = abstraction

    def get(self, entity_id: int) -> Abstraction | None:
        """Return abstraction regardless of confidence."""
        return self._store.get(entity_id, None)

    def get_active(self, entity_id: int) -> Abstraction | None:
        """Return abstraction only if it is above MIN_CONFIDENCE."""
        abs_ = self._store.get(entity_id, None)
        if abs_ and abs_.is_active:
            return abs_
        return None

    def load_seeds(self) -> None:
        """Populate store with all seed abstractions."""
        for abs_ in SEED_ABSTRACTIONS.values():
            self.register(abs_)

    # ── Queries ───────────────────────────────
    def active_all(self) -> list[Abstraction]:
        return [a for a in self._store.values() if a.is_active]

    def quarantined_all(self) -> list[Abstraction]:
        return [a for a in self._store.values() if not a.is_active]

    def size(self) -> int:
        return len(self._store)

    # ── Serialization ─────────────────────────
    def snapshot(self) -> dict:
        return {str(k): v.to_dict() for k, v in self._store.items()}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AbstractionStore":
        store = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for v in data.values():
            store.register(Abstraction.from_dict(v))
        return store

    def __repr__(self) -> str:
        active = len(self.active_all())
        quarantined = len(self.quarantined_all())
        return f"AbstractionStore(active={active}, quarantined={quarantined})"


# ─────────────────────────────────────────────
#  SEED ABSTRACTIONS — AIRIS Environment
#  Generation 0 | Confidence 1.0 | Source: seed
#  These are what the agent KNOWS before any experiment.
#  They map directly to AIRIS grid entity IDs.
# ─────────────────────────────────────────────
SEED_ABSTRACTIONS: dict[int, Abstraction] = {

    0: Abstraction(
        entity_id=0,
        name="floor",
        properties={
            "passable"    : True,
            "dangerous"   : False,
            "interactive" : False,
            "solid"       : False,
        },
        behavior=["allows_movement", "neutral_surface"],
        confidence=1.0, generation=0, source="seed", verified=True
    ),

    1: Abstraction(
        entity_id=1,
        name="agent",
        properties={
            "passable"  : False,
            "is_self"   : True,
            "movable"   : True,
        },
        behavior=["moves_in_4_directions", "collects_items", "avoids_danger"],
        confidence=1.0, generation=0, source="seed", verified=True
    ),

    2: Abstraction(
        entity_id=2,
        name="wall",
        properties={
            "passable"      : False,
            "dangerous"     : False,
            "solid"         : True,
            "destructible"  : False,
        },
        behavior=["blocks_movement", "reflects_force", "permanent_barrier"],
        confidence=1.0, generation=0, source="seed", verified=True
    ),

    3: Abstraction(
        entity_id=3,
        name="battery",
        properties={
            "passable"    : True,
            "collectable" : True,
            "goal"        : True,
            "dangerous"   : False,
        },
        behavior=["increases_score", "disappears_on_contact", "primary_objective"],
        confidence=1.0, generation=0, source="seed", verified=True
    ),

    4: Abstraction(
        entity_id=4,
        name="door",
        properties={
            "passable"  : False,
            "requires"  : "key",
            "dangerous" : False,
            "openable"  : True,
        },
        behavior=["blocks_until_key", "opens_permanently_with_key"],
        confidence=1.0, generation=0, source="seed", verified=True
    ),

    5: Abstraction(
        entity_id=5,
        name="key",
        properties={
            "passable"    : True,
            "collectable" : True,
            "goal"        : False,
            "enables"     : "door",
        },
        behavior=["adds_to_inventory", "enables_door_passage"],
        confidence=1.0, generation=0, source="seed", verified=True
    ),

    7: Abstraction(
        entity_id=7,
        name="fire",
        properties={
            "passable"  : False,
            "dangerous" : True,
            "fatal"     : True,
            "spread"    : False,
        },
        behavior=["terminates_episode", "must_avoid_or_extinguish"],
        confidence=1.0, generation=0, source="seed", verified=True
    ),

    9: Abstraction(
        entity_id=9,
        name="extinguisher",
        properties={
            "passable"    : True,
            "collectable" : True,
            "dangerous"   : False,
            "neutralizes" : "fire",
        },
        behavior=["adds_to_inventory", "neutralizes_fire_when_used"],
        confidence=1.0, generation=0, source="seed", verified=True
    ),
}
