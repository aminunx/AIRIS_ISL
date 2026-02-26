"""
test_abstractions.py
====================
AIRIS-ISL — Test Suite for abstractions.py

Every test maps to a Rule or Definition in the PRD.
Run: pytest tests/test_abstractions.py -v

Author: A.M. Almurish
Project: AIRIS-ISL
"""

import pytest
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abstractions import (
    Abstraction,
    AbstractionStore,
    SEED_ABSTRACTIONS,
    MIN_CONFIDENCE,
    CONFIDENCE_DECAY,
    VERIFY_BOOST,
    VERIFY_PENALTY,
    MAX_GENERATION,
)


# ═════════════════════════════════════════════
#  SECTION 1 — Abstraction Class
# ═════════════════════════════════════════════

class TestAbstractionCreation:

    def test_seed_floor_created_correctly(self):
        """Seed abstraction 'floor' must have correct properties."""
        floor = SEED_ABSTRACTIONS[0]
        assert floor.name == "floor"
        assert floor.properties["passable"] is True
        assert floor.properties["dangerous"] is False
        assert floor.confidence == 1.0
        assert floor.generation == 0
        assert floor.source == "seed"
        assert floor.verified is True

    def test_seed_wall_is_not_passable(self):
        """Wall must block movement — core physics rule."""
        wall = SEED_ABSTRACTIONS[2]
        assert wall.is_passable is False
        assert wall.properties["solid"] is True

    def test_seed_fire_is_dangerous(self):
        """Fire must be marked dangerous and fatal."""
        fire = SEED_ABSTRACTIONS[7]
        assert fire.is_dangerous is True
        assert fire.properties["fatal"] is True
        assert fire.is_passable is False

    def test_seed_battery_is_goal(self):
        """Battery is the primary objective."""
        battery = SEED_ABSTRACTIONS[3]
        assert battery.is_goal is True
        assert battery.is_collectable is True
        assert battery.is_dangerous is False

    def test_invalid_confidence_raises(self):
        """Confidence outside [0,1] must raise ValueError."""
        with pytest.raises(ValueError, match="confidence"):
            Abstraction(entity_id=99, name="test",
                        properties={}, behavior=[], confidence=1.5)

    def test_invalid_generation_raises(self):
        """Negative generation must raise ValueError."""
        with pytest.raises(ValueError, match="generation"):
            Abstraction(entity_id=99, name="test",
                        properties={}, behavior=[], generation=-1)

    def test_invalid_source_raises(self):
        """Unknown source must raise ValueError."""
        with pytest.raises(ValueError, match="source"):
            Abstraction(entity_id=99, name="test",
                        properties={}, behavior=[], source="unknown")

    def test_all_8_seeds_exist(self):
        """All 8 AIRIS entity types must have seed abstractions."""
        expected_ids = {0, 1, 2, 3, 4, 5, 7, 9}
        assert set(SEED_ABSTRACTIONS.keys()) == expected_ids

    def test_all_seeds_are_generation_zero(self):
        """Every seed abstraction must be generation 0."""
        for abs_ in SEED_ABSTRACTIONS.values():
            assert abs_.generation == 0, f"{abs_.name} should be gen 0"

    def test_all_seeds_confidence_is_one(self):
        """Every seed abstraction must start with confidence 1.0."""
        for abs_ in SEED_ABSTRACTIONS.values():
            assert abs_.confidence == 1.0, f"{abs_.name} should have conf 1.0"


# ═════════════════════════════════════════════
#  SECTION 2 — Confidence Updates (Rules 3 & 4)
# ═════════════════════════════════════════════

class TestConfidenceUpdates:

    def test_verify_success_increases_confidence(self):
        """PRD Rule: successful verification boosts confidence."""
        abs_ = Abstraction(entity_id=0, name="floor",
                           properties={"passable": True}, behavior=[],
                           confidence=0.80)
        abs_.verify_success()
        assert abs_.confidence == pytest.approx(0.80 + VERIFY_BOOST)

    def test_verify_failure_decreases_confidence(self):
        """PRD Rule: failed verification penalises confidence."""
        abs_ = Abstraction(entity_id=2, name="wall",
                           properties={"passable": False}, behavior=[],
                           confidence=0.80)
        abs_.verify_failure()
        assert abs_.confidence == pytest.approx(0.80 - VERIFY_PENALTY)

    def test_confidence_never_exceeds_one(self):
        """Confidence must be capped at 1.0."""
        abs_ = Abstraction(entity_id=0, name="floor",
                           properties={}, behavior=[], confidence=0.98)
        abs_.verify_success()
        assert abs_.confidence <= 1.0

    def test_confidence_never_below_zero(self):
        """Confidence must not go negative."""
        abs_ = Abstraction(entity_id=0, name="floor",
                           properties={}, behavior=[], confidence=0.05)
        abs_.verify_failure()
        assert abs_.confidence >= 0.0

    def test_quarantine_threshold(self):
        """Abstraction below MIN_CONFIDENCE must not be active."""
        abs_ = Abstraction(entity_id=0, name="floor",
                           properties={}, behavior=[],
                           confidence=MIN_CONFIDENCE - 0.01)
        assert abs_.is_active is False

    def test_active_above_threshold(self):
        """Abstraction at exactly MIN_CONFIDENCE must be active."""
        abs_ = Abstraction(entity_id=0, name="floor",
                           properties={}, behavior=[],
                           confidence=MIN_CONFIDENCE)
        assert abs_.is_active is True

    def test_correct_updates_properties_and_penalises(self):
        """PRD Rule 3: correct() updates properties but keeps the abstraction."""
        abs_ = Abstraction(entity_id=4, name="door",
                           properties={"passable": False}, behavior=[],
                           confidence=0.80)
        abs_.correct({"passable": True}, ["opens_without_key"])
        assert abs_.properties["passable"] is True
        assert "opens_without_key" in abs_.behavior
        assert abs_.confidence == pytest.approx(0.80 - VERIFY_PENALTY)

    def test_correct_never_deletes_abstraction(self):
        """PRD Rule 3: abstraction must survive after correction."""
        abs_ = Abstraction(entity_id=7, name="fire",
                           properties={"dangerous": True}, behavior=[],
                           confidence=0.40)
        abs_.verify_failure()
        # must still exist, just lower confidence
        assert abs_ is not None
        assert abs_.name == "fire"


# ═════════════════════════════════════════════
#  SECTION 3 — Generation & Derivation (Rule 4)
# ═════════════════════════════════════════════

class TestDerivation:

    def test_derive_increments_generation(self):
        """Derived abstraction must be one generation deeper."""
        parent = SEED_ABSTRACTIONS[0]  # floor, gen=0
        child = parent.derive({"slippery": True}, source="dream")
        assert child.generation == 1

    def test_derive_reduces_confidence(self):
        """PRD Rule 4: confidence decays per generation."""
        parent = SEED_ABSTRACTIONS[2]  # wall, gen=0, conf=1.0
        child = parent.derive({}, source="dream")
        expected_conf = 1.0 - (CONFIDENCE_DECAY * 1)
        assert child.confidence == pytest.approx(expected_conf)

    def test_derive_marks_unverified(self):
        """Derived abstractions start as unverified."""
        parent = SEED_ABSTRACTIONS[3]  # battery
        child = parent.derive({"magnetic": True}, source="dream")
        assert child.verified is False

    def test_generation_capped_at_max(self):
        """Generation must not exceed MAX_GENERATION."""
        abs_ = Abstraction(entity_id=0, name="floor",
                           properties={}, behavior=[],
                           confidence=0.40, generation=MAX_GENERATION)
        child = abs_.derive({}, source="dream")
        assert child.generation == MAX_GENERATION

    def test_derive_preserves_parent_properties(self):
        """Derived abstraction inherits parent properties."""
        parent = SEED_ABSTRACTIONS[2]  # wall
        child = parent.derive({"color": "red"}, source="dream")
        assert child.properties["passable"] is False  # inherited
        assert child.properties["color"] == "red"     # new


# ═════════════════════════════════════════════
#  SECTION 4 — AbstractionStore
# ═════════════════════════════════════════════

class TestAbstractionStore:

    def test_load_seeds_populates_store(self):
        """Store must contain all 8 seeds after load_seeds()."""
        store = AbstractionStore()
        store.load_seeds()
        assert store.size() == 8

    def test_get_returns_correct_abstraction(self):
        """get(entity_id) must return matching abstraction."""
        store = AbstractionStore()
        store.load_seeds()
        wall = store.get(2)
        assert wall is not None
        assert wall.name == "wall"

    def test_get_unknown_id_returns_none(self):
        """get() on unknown ID must return None."""
        store = AbstractionStore()
        store.load_seeds()
        assert store.get(999) is None

    def test_get_active_filters_quarantined(self):
        """get_active() must return None for quarantined abstractions."""
        store = AbstractionStore()
        store.load_seeds()
        # manually quarantine fire
        store.get(7).confidence = MIN_CONFIDENCE - 0.01
        result = store.get_active(7)
        assert result is None

    def test_active_all_excludes_quarantined(self):
        """active_all() must only return abstractions above threshold."""
        store = AbstractionStore()
        store.load_seeds()
        store.get(7).confidence = 0.10  # quarantine fire
        active = store.active_all()
        names = [a.name for a in active]
        assert "fire" not in names

    def test_quarantined_all_includes_low_confidence(self):
        """quarantined_all() must include abstractions below threshold."""
        store = AbstractionStore()
        store.load_seeds()
        store.get(7).confidence = 0.10
        quarantined = store.quarantined_all()
        names = [a.name for a in quarantined]
        assert "fire" in names

    def test_save_and_load_roundtrip(self, tmp_path):
        """Store must survive a save/load cycle with identical data."""
        store = AbstractionStore()
        store.load_seeds()
        path = str(tmp_path / "store.json")
        store.save(path)

        loaded = AbstractionStore.load(path)
        assert loaded.size() == store.size()
        wall_original = store.get(2).to_dict()
        wall_loaded   = loaded.get(2).to_dict()
        assert wall_original == wall_loaded

    def test_register_overwrites_existing(self):
        """register() with same entity_id must overwrite silently."""
        store = AbstractionStore()
        store.load_seeds()
        new_wall = Abstraction(
            entity_id=2, name="wall",
            properties={"passable": True},  # wrong — but testing overwrite
            behavior=[], confidence=0.50
        )
        store.register(new_wall)
        assert store.get(2).confidence == 0.50


# ═════════════════════════════════════════════
#  SECTION 5 — Serialization
# ═════════════════════════════════════════════

class TestSerialization:

    def test_to_dict_contains_all_fields(self):
        """to_dict() must include every required field."""
        abs_ = SEED_ABSTRACTIONS[3]
        d = abs_.to_dict()
        required = {"entity_id","name","properties","behavior",
                    "confidence","generation","source","verified"}
        assert required.issubset(set(d.keys()))

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict()) must produce identical abstraction."""
        original = SEED_ABSTRACTIONS[5]  # key
        restored = Abstraction.from_dict(original.to_dict())
        assert restored.name       == original.name
        assert restored.confidence == original.confidence
        assert restored.generation == original.generation
        assert restored.properties == original.properties
        assert restored.behavior   == original.behavior
