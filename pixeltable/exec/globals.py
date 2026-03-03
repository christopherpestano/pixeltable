"""
Global constants and data structures for cell materialization and reconstruction.

This module defines the sentinel key and metadata structure used to track objects
(arrays, images, binary data) that are stored outside the database in local files.
When a JSON value contains large arrays or images, CellMaterializationNode writes
them to sidecar files and replaces them with a dict containing INLINED_OBJECT_MD_KEY.
CellReconstructionNode reverses this process on read.
"""

from __future__ import annotations

import dataclasses

from pixeltable.exprs import ArrayMd, BinaryMd
from pixeltable.utils.misc import non_none_dict_factory

# Sentinel key placed in JSON dicts to mark values that were offloaded to external files.
# CellReconstructionNode looks for this key to know when to read data back from disk.
INLINED_OBJECT_MD_KEY = '__pxtinlinedobjmd__'


@dataclasses.dataclass
class InlinedObjectMd:
    """Metadata describing a single object that was inlined (offloaded) to an external file.

    When large arrays, images, or binary blobs are encountered inside a JSON value,
    CellMaterializationNode writes them to a sidecar file and stores this metadata
    in the JSON in their place. The metadata records the object type, which file
    it was written to (url_idx into the cell's file_urls list), and the byte offsets
    needed to read it back.

    Attributes:
        type: The Pixeltable column type name (e.g., 'ARRAY', 'IMAGE', 'BINARY').
        url_idx: Index into the CellMd.file_urls list identifying which file holds this object.
        img_start: Start byte offset for image data within the file.
        img_end: End byte offset for image data within the file.
        array_md: Metadata for array objects (byte offsets, bool packing info, shape).
        binary_md: Metadata for binary objects (byte offsets).
    """

    type: str  # corresponds to ts.ColumnType.Type
    url_idx: int
    img_start: int | None = None
    img_end: int | None = None
    array_md: ArrayMd | None = None
    binary_md: BinaryMd | None = None

    @classmethod
    def from_dict(cls, d: dict) -> InlinedObjectMd:
        d = d.copy()
        if 'array_md' in d:
            d['array_md'] = ArrayMd(**d['array_md'])
        if 'binary_md' in d:
            d['binary_md'] = BinaryMd(**d['binary_md'])
        return cls(**d)

    def as_dict(self) -> dict:
        result = dataclasses.asdict(self, dict_factory=non_none_dict_factory)
        if self.array_md is not None:
            result['array_md'] = self.array_md.as_dict()
        if self.binary_md is not None:
            result['binary_md'] = dataclasses.asdict(self.binary_md)
        return result
