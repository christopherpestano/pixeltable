"""Dating-profile sample app for the Pixeltable examples gallery.

This package contains a small end-to-end demo of building a dating-profile
generator on top of Pixeltable:

* photo ingestion and captioning (Tasks A-C, see the KAN-43 epic),
* top-photo selection with diversity constraints (Task D),
* LLM bio generation from the selected photos (this module, ``bio``),
* a small CLI / smoke test driver (Task F).

Only ``bio`` is required to stand on its own; it accepts the ``list[dict]``
hand-off shape produced by the photo-selection step and can be exercised
without a live Pixeltable database.
"""

from examples.dating_profile.bio import generate_bio

__all__ = ['generate_bio']
