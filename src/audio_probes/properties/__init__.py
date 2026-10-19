"""Property registry. Adding a property = write one module and add it here."""

from __future__ import annotations

import importlib

from audio_probes.properties.base import Property

_REGISTRY = {
    "pitch": "audio_probes.properties.pitch:Pitch",
    "highband": "audio_probes.properties.highband:HighBand",
    "distance": "audio_probes.properties.distance:Distance",
}

PROPERTY_NAMES = list(_REGISTRY)


def get_property(cfg: dict) -> Property:
    module_name, class_name = _REGISTRY[cfg["property"]].split(":")
    return getattr(importlib.import_module(module_name), class_name)(cfg)
