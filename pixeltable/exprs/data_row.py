from __future__ import annotations

import dataclasses
import datetime
import io
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pgvector.sqlalchemy  # type: ignore[import-untyped]
import PIL.Image
import sqlalchemy as sql

import pixeltable.utils.image as image_utils
from pixeltable import catalog, env
from pixeltable.utils.local_store import TempStore
from pixeltable.utils.misc import non_none_dict_factory


@dataclasses.dataclass
class ArrayMd:
    """
    Metadata for array cells that are stored externally.

    When arrays are too large to store inline in the database, they are written to external storage.
    This metadata tracks the byte range within the external file where the array data resides.
    """

    # Byte offset where array data begins in external storage
    start: int
    # Byte offset where array data ends in external storage
    end: int

    # Boolean arrays are stored as packed bits (uint8) for efficiency; shape is needed to reconstruct
    is_bool: bool = False
    shape: tuple[int, ...] | None = None

    def as_dict(self) -> dict:
        # dict_factory: suppress Nones
        x = dataclasses.asdict(self, dict_factory=non_none_dict_factory)
        return x


@dataclasses.dataclass
class BinaryMd:
    """
    Metadata for binary cells that are stored externally.

    Similar to ArrayMd, this tracks byte ranges for binary data stored outside the database.
    """

    # Byte offset where binary data begins in external storage
    start: int
    # Byte offset where binary data ends in external storage
    end: int


@dataclasses.dataclass
class CellMd:
    """
    Content of the cellmd column - per-cell metadata stored alongside values.

    This metadata column stores auxiliary information about a cell's value:
    - Error information when expression evaluation fails
    - References to external files for media/array data
    - Byte offsets for externally stored arrays/binaries

    All fields are optional to minimize storage overhead for the common case.
    """

    # Exception class name when evaluation failed (e.g., "ValueError")
    errortype: str | None = None
    # Exception message when evaluation failed
    errormsg: str | None = None

    # URLs referencing external files containing cell data:
    # - For json columns: list of all media URLs embedded in the JSON structure
    # - For array columns: single URL pointing to the serialized array file
    file_urls: list[str] | None = None

    # Byte range metadata for externally stored arrays
    array_md: ArrayMd | None = None
    # Byte range metadata for externally stored binary data
    binary_md: BinaryMd | None = None

    @classmethod
    def from_dict(cls, d: dict) -> CellMd:
        d = d.copy()
        if 'array_md' in d:
            d['array_md'] = ArrayMd(**d['array_md'])
        if 'binary_md' in d:
            d['binary_md'] = BinaryMd(**d['binary_md'])
        return cls(**d)

    def as_dict(self) -> dict:
        x = dataclasses.asdict(self, dict_factory=non_none_dict_factory)
        return x


class DataRow:
    """
    Encapsulates all data and execution state needed by RowBuilder and DataRowBatch:
    - state for in-memory computation
    - state needed for expression evaluation
    - containers for output column values

    This is not meant to be a black-box abstraction.

    In-memory representations by column type:
    - StringType: str
    - IntType: int
    - FloatType: float
    - BoolType: bool
    - TimestampType: datetime.datetime
    - DateType: datetime.date
    - UUIDType: uuid.UUID
    - BinaryType: bytes
    - JsonType: json-serializable object
    - ArrayType: numpy.ndarray
    - ImageType: PIL.Image.Image
    - VideoType: local path if available, otherwise url
    - AudioType: local path if available, otherwise url
    - DocumentType: local path if available, otherwise url
    """

    # === Expression evaluation state (indexed by slot_idx) ===
    # Each expression in the DAG is assigned a unique slot_idx; these arrays track per-slot state.

    # Computed values for each expression slot
    vals: np.ndarray  # of object
    # True if slot has been computed (may still be None if computation returned None or raised exception)
    has_val: np.ndarray  # of bool
    # Exception objects for slots where evaluation failed
    excs: np.ndarray  # of object
    # Boolean mask of slots that still need to be computed for this row
    missing_slots: np.ndarray  # of bool; number of missing dependencies
    # Count of dependent slots that haven't been computed yet (used for garbage collection)
    missing_dependents: np.ndarray  # of int16; number of missing dependents
    # Prevents duplicate scheduling: True if slot evaluation has already been queued
    is_scheduled: np.ndarray  # of bool; True if this slot is scheduled for evaluation

    # CellMd needed for query execution; needs to be indexed by slot idx, not column id, to work for joins
    slot_md: dict[int, CellMd]

    # file_urls:
    # - stored url of file for media in vals[i]
    # - None if vals[i] is not media type
    # - not None if file_paths[i] is not None
    # TODO: this is a sparse vector; should it be a dict[int, str]?
    file_urls: np.ndarray  # of str

    # file_paths:
    # - local path of media file in vals[i]; points to the file cache if file_urls[i] is remote
    # - None if vals[i] is not a media type or if there is no local file yet for file_urls[i]
    # TODO: this is a sparse vector; should it be a dict[int, str]?
    file_paths: np.ndarray  # of str

    # If `may_have_exc` is False, then we guarantee that no slot has an exception set. This is used to optimize
    # exception handling under normal operation.
    _may_have_exc: bool

    # the primary key of a store row is a sequence of ints (the number is different for table vs view)
    pk: tuple[int, ...] | None
    # for nested rows (ie, those produced by JsonMapperDispatcher)
    parent_row: DataRow | None
    parent_slot_idx: int | None

    # state for table output (insert()/update()); key: column id
    cell_vals: dict[int, Any]  # materialized values of output columns, in the format required for the column
    cell_md: dict[int, CellMd]

    # === Slot type indices (shared across all DataRows in a batch) ===
    # These lists identify which slots contain specific data types, enabling type-specific handling.
    # Images are loaded lazily from file_paths when accessed via __getitem__
    img_slot_idxs: list[int]
    # Non-image media (video, audio, document) - stored as file paths/URLs
    media_slot_idxs: list[int]
    # NumPy arrays - serialized to bytes for storage
    array_slot_idxs: list[int]
    # JSON data - requires special NULL handling for storage
    json_slot_idxs: list[int]

    def __init__(
        self,
        size: int,
        img_slot_idxs: list[int],
        media_slot_idxs: list[int],
        array_slot_idxs: list[int],
        json_slot_idxs: list[int],
        parent_row: DataRow | None = None,
        parent_slot_idx: int | None = None,
    ):
        self.init(size)
        self.parent_row = parent_row
        self.parent_slot_idx = parent_slot_idx
        self.img_slot_idxs = img_slot_idxs
        self.media_slot_idxs = media_slot_idxs
        self.array_slot_idxs = array_slot_idxs
        self.json_slot_idxs = json_slot_idxs

    def init(self, size: int) -> None:
        """Initialize or reset all slot arrays to their default state.

        Args:
            size: Number of slots (expressions) this row needs to hold.
        """
        self.vals = np.full(size, None, dtype=object)
        self.has_val = np.zeros(size, dtype=bool)
        self.excs = np.full(size, None, dtype=object)
        self.missing_slots = np.zeros(size, dtype=bool)
        self.missing_dependents = np.zeros(size, dtype=np.int16)
        self.is_scheduled = np.zeros(size, dtype=bool)
        self.slot_md = {}
        self.file_urls = np.full(size, None, dtype=object)
        self.file_paths = np.full(size, None, dtype=object)
        self._may_have_exc = False
        self.cell_vals = {}
        self.cell_md = {}
        self.pk = None
        self.parent_row = None
        self.parent_slot_idx = None

    def clear(self, slot_idxs: np.ndarray | None = None) -> None:
        """Clear computed values for specified slots (garbage collection) or reset entire row.

        Called during expression evaluation to free memory for intermediate results
        that are no longer needed after their dependents have been computed.

        Args:
            slot_idxs: Boolean mask or index array of slots to clear. If None, resets entire row.
        """
        if slot_idxs is not None:
            self.has_val[slot_idxs] = False
            self.vals[slot_idxs] = None
            self.excs[slot_idxs] = None
            self.file_urls[slot_idxs] = None
            self.file_paths[slot_idxs] = None
        else:
            self.init(len(self.vals))

    def set_file_path(self, idx: int, path: str) -> None:
        """Augment an existing url with a local file path"""
        assert self.has_val[idx]
        assert idx in self.img_slot_idxs or idx in self.media_slot_idxs
        self.file_paths[idx] = path
        if idx in self.media_slot_idxs:
            self.vals[idx] = path

    def copy(self, target: DataRow) -> None:
        """Create a copy of the contents of this DataRow in target
        The copy shares the cell values, but not the control structures (eg, self.has_val), because these
        need to be independently updateable.
        """
        target.vals = self.vals.copy()
        target.has_val = self.has_val.copy()
        target.excs = self.excs.copy()
        target.pk = self.pk
        target.file_urls = self.file_urls.copy()
        target.file_paths = self.file_paths.copy()

    def set_pk(self, pk: tuple[int, ...]) -> None:
        self.pk = pk

    def has_exc(self, slot_idx: int | None = None) -> bool:
        """
        Returns True if an exception has been set for the given slot index, or for any slot index if slot_idx is None
        """
        if not self._may_have_exc:
            return False

        if slot_idx is not None:
            return self.excs[slot_idx] is not None
        return (self.excs != None).any()

    def get_exc(self, slot_idx: int) -> Exception | None:
        exc = self.excs[slot_idx]
        assert exc is None or isinstance(exc, Exception)
        return exc

    def get_first_exc(self) -> Exception | None:
        mask = self.excs != None
        if not mask.any():
            return None
        return self.excs[mask][0]

    def set_exc(self, slot_idx: int, exc: Exception) -> None:
        """Record an exception for a slot that failed during evaluation.

        The slot is marked as having a value (None) to prevent re-evaluation.
        Exceptions propagate to dependent slots via RowBuilder.set_exc().
        """
        assert self.excs[slot_idx] is None
        self.excs[slot_idx] = exc
        self._may_have_exc = True

        # Mark as computed with None value - prevents re-evaluation attempts
        self.has_val[slot_idx] = True
        self.vals[slot_idx] = None
        self.file_paths[slot_idx] = None
        self.file_urls[slot_idx] = None

    def __getitem__(self, index: int) -> Any:
        """Returns in-memory value, ie, what is needed for expr evaluation"""
        assert isinstance(index, int)
        if not self.has_val[index]:
            # This is a sufficiently cheap and sensitive validation that it makes sense to keep the assertion around
            # even if python is running with -O.
            raise AssertionError(index)

        if self.file_urls[index] is not None and index in self.img_slot_idxs:
            # if we need to load this from a file, it should have been materialized locally
            # TODO this fails if the url was instantiated dynamically using astype()
            assert self.file_paths[index] is not None
            if self.vals[index] is None:
                self.vals[index] = PIL.Image.open(self.file_paths[index])
                self.vals[index].load()

        return self.vals[index]

    def get_stored_val(self, index: int, sa_col_type: sql.types.TypeEngine | None = None) -> Any:
        """Return the value that gets stored in the db"""
        assert self.excs[index] is None
        if not self.has_val[index]:
            # for debugging purposes
            pass
        assert self.has_val[index]

        if self.file_urls[index] is not None and (index in self.img_slot_idxs or index in self.media_slot_idxs):
            # if this is an image or other media type we want to store, we should have a url
            return self.file_urls[index]

        if self.vals[index] is not None and index in self.array_slot_idxs:
            assert isinstance(self.vals[index], np.ndarray)
            np_array = self.vals[index]
            if sa_col_type is not None and isinstance(
                sa_col_type, (pgvector.sqlalchemy.Vector, pgvector.sqlalchemy.HALFVEC)
            ):
                return np_array
            buffer = io.BytesIO()
            np.save(buffer, np_array)
            return buffer.getvalue()

        # for JSON columns, we need to store None as an explicit NULL, otherwise it stores a json 'null'
        if self.vals[index] is None and sa_col_type is not None and isinstance(sa_col_type, sql.JSON):
            return sql.sql.null()

        if isinstance(self.vals[index], datetime.datetime) and self.vals[index].tzinfo is None:
            # if the datetime is naive, cast it to the default time zone
            return self.vals[index].replace(tzinfo=env.Env.get().default_time_zone)

        return self.vals[index]

    def __setitem__(self, idx: int, val: Any) -> None:
        """Assign in-memory cell value with type-specific handling.

        Media values (str paths/URLs) are parsed to populate file_urls/file_paths.
        Array values (bytes) are deserialized from NumPy format.
        This allows overwriting existing values.
        """
        assert isinstance(idx, int)
        assert self.excs[idx] is None

        if (idx in self.img_slot_idxs or idx in self.media_slot_idxs) and isinstance(val, str):
            # this is either a local file path or a URL
            parsed = urllib.parse.urlparse(val)
            # Determine if this is a local file or a remote URL. If the scheme length is <= 1,
            # we assume it's a local file. (This is because a Windows path will be interpreted
            # by urllib as a URL with scheme equal to the drive letter.)
            if len(parsed.scheme) <= 1 or parsed.scheme == 'file':
                # local file path
                assert self.file_urls[idx] is None and self.file_paths[idx] is None
                if len(parsed.scheme) <= 1:
                    path = str(Path(val).absolute())  # Ensure we're using an absolute pathname.
                    self.file_urls[idx] = urllib.parse.urljoin('file:', urllib.request.pathname2url(path))
                    self.file_paths[idx] = path
                else:  # file:// URL
                    self.file_urls[idx] = val
                    # Wrap the path in a url2pathname() call to ensure proper handling on Windows.
                    self.file_paths[idx] = urllib.parse.unquote(urllib.request.url2pathname(parsed.path))
            else:
                # URL
                assert self.file_urls[idx] is None
                self.file_urls[idx] = val

            if idx in self.media_slot_idxs:
                self.vals[idx] = self.file_paths[idx] if self.file_paths[idx] is not None else self.file_urls[idx]
        elif idx in self.array_slot_idxs and isinstance(val, bytes):
            self.vals[idx] = np.load(io.BytesIO(val))
        else:
            self.vals[idx] = val
        self.has_val[idx] = True

    def prepare_col_val_for_save(self, index: int, col: catalog.Column | None = None) -> bool:
        """
        Prepare to save a column's value into the appropriate store. Discard unneeded values.

        Return:
            True if the media object in the column needs to be saved.
        """
        if self.vals[index] is None:
            return False

        if self.file_urls[index] is not None:
            return False

        assert self.excs[index] is None
        if self.file_paths[index] is None:
            if col is not None:
                # This is a media object that needs to be saved
                return True
            else:
                # This is a media object that we don't care about, so we discard it
                self.has_val[index] = False
        else:
            # we already have a file for this image, nothing left to do
            pass

        self.vals[index] = None
        return False

    def save_media_to_temp(self, index: int, col: catalog.Column) -> str:
        """Save the media object in the column to the TempStore.
        Objects cannot be saved directly to general destinations."""
        assert col.col_type.is_media_type()
        val = self.vals[index]
        format = None
        if isinstance(val, PIL.Image.Image):
            format = image_utils.default_format(val)
        filepath, url = TempStore.save_media_object(val, col, format=format)
        self.file_paths[index] = str(filepath) if filepath is not None else None
        self.vals[index] = None
        return url

    @property
    def rowid(self) -> tuple[int, ...]:
        """The row identifier portion of the primary key (excludes version number)."""
        return self.pk[:-1]

    @property
    def v_min(self) -> int:
        """The version number from the primary key (last element)."""
        return self.pk[-1]
