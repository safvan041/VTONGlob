"""Registry mapping jewelry type strings to engine functions.

The web view imports this registry to dynamically dispatch the correct
try‑on implementation based on the user selection.
"""

from .earring_engine import tryon_earring
from .nosering_engine import tryon_nosering
from .necklace_engine import tryon_necklace

# Mapping used by the view. Keys must match the values displayed in the HTML
# dropdown.
REGISTRY = {
    "earring": tryon_earring,
    "nose": tryon_nosering,  # displayed as "nose ring"
    "necklace": tryon_necklace,
}
