"""Conversion of pandas DataFrames to values of FlodymArrays (private module).

The layout of the DataFrame follows from its column names, without looking at the entries:

- Dimension columns are named by dimension name or letter. Dimensions with one item need no
  column.
- Long format: there is a column ``value``, or every dimension has a column. The value column is
  ``value``, or else the only further column. Other columns are ignored.
- Wide format: there is no column ``value``, exactly one dimension has no column, and some column
  names are items of it. These columns hold the values. Further columns without numbers, e.g.
  units or comments, are ignored.

The entries are only used for the hints in error messages.
"""

import itertools
import logging
import re
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ._base_model import (
    MAX_LISTED_VALUES,
    FlodymDataError,
    closest_name,
    list_values,
    to_builtin,
)
from .dimensions import Dimension, DimensionSet

if TYPE_CHECKING:
    from .flodym_arrays import FlodymArray

logger = logging.getLogger(__name__)

MAX_LISTED_COMBINATIONS = 10
"""Maximum number of missing index combinations listed in an error message."""

VALUE_COLUMN = "value"
"""Name of the value column. A column of this name means long format."""

NOT_AN_ITEM = object()
"""Returned by :py:func:`as_item` for values which are not an item of the dimension."""

UNNAMED_COLUMN = re.compile(r"^Unnamed: \d+$")
"""Name pandas gives to columns with an empty header cell."""

RENAMED_DUPLICATE = re.compile(r"^(.+)\.\d+$")
"""Name pandas gives to the second and further columns with the same header, e.g. ``2020.1``."""


@dataclass
class DataLayout:
    """How the columns of a DataFrame are mapped to the dimensions and values of an array."""

    dim_columns: dict[str, Hashable]
    """Column of each dimension that has one, by dimension name."""
    reason: str
    """Why this layout was chosen, for error messages."""
    value_column: Hashable | None = None
    """Long format: the column with the values."""
    wide_dim: Dimension | None = None
    """Wide format: the dimension whose items name the value columns."""
    item_columns: dict[Hashable, Any] = field(default_factory=dict)
    """Wide format: the item of :py:attr:`wide_dim` for each value column."""
    ignored_columns: list[Hashable] = field(default_factory=list)
    """Columns which are neither dimension nor value columns."""


class DataFrameToFlodymDataConverter:
    """Converts a pandas DataFrame to a numpy array that can be used as values of a FlodymArray.

    Usually not called by the user, but from within the FlodymArray from_df and
    set_values_from_df methods, where further documentation can be found. The layout rule is
    described in the docstring of this module.

    Problems with the data raise a :py:class:`flodym.FlodymDataError`. Where single entries are
    the problem, the message gives their labels in the index of the DataFrame.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        flodym_array: "FlodymArray",
        allow_missing_values: bool = False,
        allow_extra_values: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        self.df = df.copy()
        self.flodym_array = flodym_array
        self.dims: DimensionSet = flodym_array.dims
        self.allow_missing_values = allow_missing_values
        self.allow_extra_values = allow_extra_values
        self.strip_whitespace = strip_whitespace
        try:
            self.target_values = self._get_target_values()
        except FlodymDataError as error:
            error._add_context(self._error_context)
            raise
        except Exception as error:  # re-raised below with added context
            wrapped = FlodymDataError(f"{type(error).__name__}: {error}")
            wrapped._add_context(self._error_context)
            raise wrapped from error

    @property
    def _error_context(self) -> str:
        """First line of error messages, naming the FlodymArray whose values are set."""
        array_type = type(self.flodym_array).__name__
        dims_text = ", ".join(f"{dim.name} ({dim.letter})" for dim in self.dims) or "no dimensions"
        name = self.flodym_array.name
        if name in (None, "unnamed"):
            return (
                f"While setting values of an unnamed {array_type} with dimensions {dims_text} "
                f"from a DataFrame (give name=... to see the name here):"
            )
        return (
            f"While setting values of {array_type} '{name}' with dimensions {dims_text} "
            "from a DataFrame:"
        )

    def _get_target_values(self) -> np.ndarray:
        """Convert the DataFrame to an array of the FlodymArray's shape.

        Returns:
            The values for the FlodymArray.
        """
        self._move_named_index_to_columns()
        if self.strip_whitespace:
            self._strip_whitespace_from_df()
        self._drop_empty_rows_and_columns()
        if not self._read_single_field():
            layout = determine_layout(
                self.df, self.dims, self.allow_extra_values, self.allow_missing_values
            )
            if layout.ignored_columns:
                logger.info(
                    f"Ignoring columns of '{self.flodym_array.name}' which are neither "
                    f"dimension nor value columns: {list_values(layout.ignored_columns)}"
                )
            self.df = to_long_format(self.df, layout)
        self._add_single_item_dims()
        self._check_empty_dim_cells()
        self._convert_type()
        self.df = self.df[list(self.dims.names) + [VALUE_COLUMN]]
        return self._check_data_complete()

    def _move_named_index_to_columns(self) -> None:
        """Make index levels with a name columns, e.g. after ``read_csv(..., index_col=[0, 1])``.

        An unnamed index is kept, as error messages use its labels to name rows.
        """
        if any(name is not None for name in self.df.index.names):
            self.df.reset_index(inplace=True)

    def _strip_whitespace_from_df(self) -> None:
        logger.debug("Stripping whitespace from string values in the DataFrame.")

        def strip_if_string(x: object) -> object:
            if isinstance(x, str):
                return x.strip()
            return x

        self.df.columns = self.df.columns.map(strip_if_string)
        if not pd.api.types.is_numeric_dtype(self.df.index.dtype):
            self.df.index = self.df.index.map(strip_if_string)
        for position, dtype in enumerate(self.df.dtypes):
            if pd.api.types.is_numeric_dtype(dtype):
                continue
            column = self.df.iloc[:, position]
            if isinstance(dtype, pd.StringDtype):
                # Perform vectorized strip for string columns for efficiency
                self.df.isetitem(position, column.str.strip())
            else:
                self.df.isetitem(position, column.map(strip_if_string))

    def _drop_empty_rows_and_columns(self) -> None:
        """Drop rows without any entry, and columns without entries and without a name.

        Such rows and columns occur e.g. in Excel sheets with formatted but empty cells.
        """
        empty_columns = [
            column
            for column in self.df.columns
            if is_unnamed(column) and self.df[column].isna().all()
        ]
        if empty_columns:
            logger.debug(f"Dropping empty columns without name: {empty_columns}")
            self.df = self.df.drop(columns=empty_columns)
        self.df = self.df.dropna(how="all")
        if len(self.df.columns) == 0:
            raise FlodymDataError(
                "The DataFrame has no columns. Check that the file or sheet contains the data."
            )

    def _read_single_field(self) -> bool:
        """Read the value of an array with a single value from a file with a single field.

        pandas reads such a file with the field as column name and no rows. This is only done if
        all dimensions have one item, as then exactly one number is needed.

        Returns:
            Whether the value was read.
        """
        if any(dim.len > 1 for dim in self.dims) or len(self.df) > 0 or len(self.df.columns) > 1:
            return False
        try:
            value = float(self.df.columns[0])
        except (TypeError, ValueError):
            return False
        self.df = pd.DataFrame({VALUE_COLUMN: [value]})
        return True

    def _add_single_item_dims(self) -> None:
        """Add a column for each dimension with one item that has no column."""
        for dim in self.dims:
            if dim.name not in self.df.columns:
                self.df[dim.name] = dim.items[0]

    def _check_empty_dim_cells(self) -> None:
        """Check for empty cells in dimension columns; drop their rows if extra values are allowed.

        Raises:
            FlodymDataError: If there are empty cells and extra values are not allowed.
        """
        for dim in self.dims:
            empty = self.df[dim.name].isna()
            if not empty.any():
                continue
            if self.allow_extra_values:
                self.df = self.df[~empty]
                continue
            raise FlodymDataError(
                f"Column '{dim.name}' has empty cells (index {self._index_labels(empty)}). Fill "
                "them, remove these rows, or set allow_extra_values=True to ignore them."
            )

    def _convert_type(self) -> None:
        """Convert the dimension columns to the dtype of their dimension, and the values to float.

        Raises:
            FlodymDataError: If an entry cannot be converted.
        """
        for dim in self.dims:
            if dim.dtype is None:
                continue
            converted: dict[Any, Any] = {}
            failed: list[Any] = []
            for item in self.df[dim.name].unique():
                try:
                    converted[item] = dim.dtype(item)
                except (ValueError, TypeError):
                    failed.append(item)
            if failed:
                raise FlodymDataError(
                    f"Column '{dim.name}' has entries that cannot be converted to the type "
                    f"{dim.dtype.__name__} of the dimension items: "
                    f"{self._entries_with_index(dim.name, failed, suggestions=dim.items)}."
                )
            self.df[dim.name] = self.df[dim.name].map(converted)
        values = pd.to_numeric(self.df[VALUE_COLUMN], errors="coerce")
        not_numbers = values.isna() & self.df[VALUE_COLUMN].notna()
        if not_numbers.any():
            entries = self.df.loc[not_numbers, VALUE_COLUMN].unique().tolist()
            raise FlodymDataError(
                "The value column has entries that are not numbers: "
                f"{self._entries_with_index(VALUE_COLUMN, entries)}."
            )
        self.df[VALUE_COLUMN] = values.astype(np.float64)

    def _check_data_complete(self) -> np.ndarray:
        # Special handling of the scalar case: If the FlodymArray has no dimensions, we usually
        # expect a single value in the df.
        if self.dims.ndim == 0:
            if len(self.df) == 0:
                if self.allow_missing_values:
                    return np.array(0.0)
                raise FlodymDataError(
                    "FlodymArray has no dimensions, but the DataFrame has no rows. Expected "
                    "exactly one row, or a single field, which pandas reads as column name."
                )
            if len(self.df) > 1:
                raise FlodymDataError(
                    f"FlodymArray has no dimensions, but the DataFrame has {len(self.df)} rows. "
                    "Expected exactly one row."
                )
            return np.array(self.df[VALUE_COLUMN].values[0])

        self._check_no_duplicates()

        # remove rows with extra values or throw error
        if self.allow_extra_values:
            for dim in self.dims:
                self.df = self.df[self.df[dim.name].isin(dim.items)]
        else:
            for dim in self.dims:
                self._check_no_extra_items(dim)

        if self.allow_missing_values:
            self.df[VALUE_COLUMN] = self.df[VALUE_COLUMN].fillna(0)
        else:
            # Duplicates throw an error, and extra values are removed above,
            # so unequal lengths indicate missing values
            if len(self.df) != self.flodym_array.size:
                self._raise_missing_values()
            self._check_no_empty_values()

        # for performance and memory reasons, we use numpy operations to convert the data to the
        # numpy array
        # replace non-value items in df with their index in dims.items
        for dim in self.dims:
            self.df[dim.name] = self.df[dim.name].map({item: i for i, item in enumerate(dim.items)})
        # convert df to numpy index array: Take all non-value columns
        fill_indices = self.df.values[:, :-1].T.astype(np.intp)
        fill_values = self.df[VALUE_COLUMN].values
        # this is what ends up in the parameter; initialize with zeros
        values = np.zeros(self.dims.shape)
        values[tuple(fill_indices)] = fill_values
        return values

    def _check_no_duplicates(self) -> None:
        """Raise an error if index combinations occur more than once."""
        indices = self.df[list(self.dims.names)]
        duplicated = indices.duplicated(keep=False)
        if not duplicated.any():
            return
        groups = indices[duplicated].groupby(list(indices.columns), sort=False).groups
        listed = [
            f"{_combination(combination)} (index {list_values(dict.fromkeys(labels))})"
            for combination, labels in itertools.islice(groups.items(), MAX_LISTED_VALUES)
        ]
        more = f"\n  ... ({len(groups)} in total)" if len(groups) > MAX_LISTED_VALUES else ""
        raise FlodymDataError(
            "These index combinations occur more than once in the data:\n  "
            + "\n  ".join(listed)
            + more
        )

    def _check_no_extra_items(self, dim: Dimension) -> None:
        """Raise an error if a dimension column contains items that are not in the dimension."""
        dim_items = set(dim.items)
        extra_items = [item for item in self.df[dim.name].unique() if item not in dim_items]
        if not extra_items:
            return
        message = (
            f"Column '{dim.name}' has items that are not in the dimension: "
            f"{self._entries_with_index(dim.name, extra_items, suggestions=dim.items)}."
        )
        type_hint = _type_hint(extra_items, dim)
        if type_hint:
            message += f"\n{type_hint}"
        message += (
            "\nCorrect these items, add them to the dimension, or set allow_extra_values=True to "
            "ignore these rows."
        )
        raise FlodymDataError(message)

    def _raise_missing_values(self) -> None:
        # print warning first, as compiling expected index tuples may take long
        logger.warning(
            f"{self._error_context} Detected missing values in the data, but "
            f"allow_missing_values is set to False. Expected {self.flodym_array.size} "
            f"rows, but only got {len(self.df)}. Computing missing values...."
        )
        indices = self.df[list(self.dims.names)]
        actual_index_tuples = set(indices.itertuples(index=False, name=None))
        missing = [
            combination
            for combination in itertools.product(*[dim.items for dim in self.dims])
            if combination not in actual_index_tuples
        ]
        listed = ", ".join(_combination(c) for c in missing[:MAX_LISTED_COMBINATIONS])
        more = ", ..." if len(missing) > MAX_LISTED_COMBINATIONS else ""
        raise FlodymDataError(
            f"Values are missing for {len(missing)} of {self.flodym_array.size} index "
            f"combinations ({', '.join(self.dims.names)}): {listed}{more}\n"
            "Add them, or set allow_missing_values=True to set them to zero."
        )

    def _check_no_empty_values(self) -> None:
        """Raise an error if values are empty, naming their index combinations and rows."""
        empty = self.df[VALUE_COLUMN].isna()
        if not empty.any():
            return
        dim_names = list(self.dims.names)
        combinations = self.df.loc[empty, dim_names].itertuples(index=True, name=None)
        listed = ", ".join(
            f"{_combination(tuple(combination[1:]))} (index {combination[0]!r})"
            for combination in itertools.islice(combinations, MAX_LISTED_VALUES)
        )
        n_empty = int(empty.sum())
        more = f", ... ({n_empty} in total)" if n_empty > MAX_LISTED_VALUES else ""
        raise FlodymDataError(
            f"Values are empty for these index combinations ({', '.join(dim_names)}): "
            f"{listed}{more}.\nFill them, or set allow_missing_values=True to set them to zero."
        )

    def _entries_with_index(
        self, column: Hashable, entries: Sequence[Any], suggestions: Iterable[Any] = ()
    ) -> str:
        """List entries of a column with the index labels of their rows, and suggestions."""
        suggestions = list(suggestions)
        listed = []
        for entry in entries[:MAX_LISTED_VALUES]:
            text = f"{to_builtin(entry)!r} (index {self._index_labels(self.df[column] == entry)}"
            suggestion = closest_name(entry, suggestions)
            if suggestion is not None:
                text += f"; did you mean {suggestion!r}?"
            listed.append(text + ")")
        more = f", ... ({len(entries)} in total)" if len(entries) > MAX_LISTED_VALUES else ""
        return ", ".join(listed) + more

    def _index_labels(self, mask: pd.Series) -> str:
        """List the index labels of the rows selected by ``mask``, each label once."""
        return list_values(dict.fromkeys(self.df.index[mask]))


def determine_layout(
    df: pd.DataFrame,
    dims: DimensionSet,
    allow_extra_values: bool = False,
    allow_missing_values: bool = False,
) -> DataLayout:
    """Determine the layout of a DataFrame from its column names, see the module docstring.

    Raises:
        FlodymDataError: If the column names do not give a valid layout.
    """
    dim_columns: dict[str, Hashable] = {}
    for dim in dims:
        columns = [c for c in df.columns if c in (dim.name, dim.letter)]
        if len(columns) > 1:
            raise FlodymDataError(
                f"Dimension '{dim.name}' has several columns: {list_values(columns)}. Give one "
                "column, named by dimension name or letter."
            )
        if columns:
            dim_columns[dim.name] = columns[0]
    other_columns = [c for c in df.columns if c not in dim_columns.values()]
    dims_without_column = [dim for dim in dims if dim.len > 1 and dim.name not in dim_columns]

    if VALUE_COLUMN in other_columns or not dims_without_column:
        if VALUE_COLUMN in other_columns:
            reason = f"Long format assumed, as there is a column '{VALUE_COLUMN}'."
        else:
            reason = "Long format assumed, as every dimension has a column."
        if dims_without_column:
            raise FlodymDataError(_layout_problem(df, dims_without_column, other_columns, reason))
        return _long_layout(dim_columns, other_columns, reason)

    if len(dims_without_column) == 1:
        wide_dim = dims_without_column[0]
        item_columns = {
            c: item
            for c in other_columns
            if not is_unnamed(c) and (item := as_item(c, wide_dim)) is not NOT_AN_ITEM
        }
        if item_columns:
            reason = (
                f"Wide format over '{wide_dim.name}' assumed, as there is no column "
                f"'{VALUE_COLUMN}' and '{wide_dim.name}' is the only dimension without column."
            )
            return _wide_layout(
                df,
                dim_columns=dim_columns,
                wide_dim=wide_dim,
                item_columns=item_columns,
                other_columns=other_columns,
                reason=reason,
                allow_extra_values=allow_extra_values,
                allow_missing_values=allow_missing_values,
            )
    raise FlodymDataError(_layout_problem(df, dims_without_column, other_columns, ""))


def _long_layout(
    dim_columns: dict[str, Hashable], other_columns: list[Hashable], reason: str
) -> DataLayout:
    """Layout with one value column: ``value``, or else the only further column with a name."""
    if VALUE_COLUMN in other_columns:
        value_column = VALUE_COLUMN
    else:
        named = [c for c in other_columns if not is_unnamed(c)]
        candidates = named if named else other_columns
        if not candidates:
            raise FlodymDataError(
                f"{reason} But there is no value column: all columns are dimension columns."
            )
        if len(candidates) > 1:
            raise FlodymDataError(
                f"{reason} But there are several further columns, so the value column is "
                f"unclear: {list_values(other_columns)}. Name the value column "
                f"'{VALUE_COLUMN}'; then further columns are ignored."
            )
        value_column = candidates[0]
    ignored = [c for c in other_columns if c != value_column]
    return DataLayout(
        dim_columns=dim_columns, reason=reason, value_column=value_column, ignored_columns=ignored
    )


def _wide_layout(
    df: pd.DataFrame,
    dim_columns: dict[str, Hashable],
    wide_dim: Dimension,
    item_columns: dict[Hashable, Any],
    other_columns: list[Hashable],
    reason: str,
    allow_extra_values: bool,
    allow_missing_values: bool,
) -> DataLayout:
    """Layout with value columns named by the items of a dimension.

    Raises:
        FlodymDataError: If further columns contain numbers and extra values are not allowed, or
            if items have no column and missing values are not allowed.
    """
    ignored = [c for c in other_columns if c not in item_columns]
    problems = []
    numeric = [c for c in ignored if not is_unnamed(c) and _contains_numbers(df[c])]
    if numeric and not allow_extra_values:
        problems.append(
            f"- Further columns contain numbers, but their names are not items of "
            f"'{wide_dim.name}': {list_values(numeric)}. Remove them, or set "
            "allow_extra_values=True to ignore them. Further columns are only ignored if they "
            "contain no numbers, e.g. units or comments."
        )
        problems += [f"  {hint}" for c in numeric if (hint := _duplicate_hint(c, wide_dim))]
    found_items = set(item_columns.values())
    missing_items = [item for item in wide_dim.items if item not in found_items]
    if missing_items and not allow_missing_values:
        problems.append(
            f"- Items of '{wide_dim.name}' without column: {list_values(missing_items)}. Add "
            "these columns, or set allow_missing_values=True to set their values to zero."
        )
    if problems:
        raise FlodymDataError("\n".join([reason, *problems]))
    return DataLayout(
        dim_columns=dim_columns,
        reason=reason,
        wide_dim=wide_dim,
        item_columns=item_columns,
        ignored_columns=ignored,
    )


def to_long_format(df: pd.DataFrame, layout: DataLayout) -> pd.DataFrame:
    """Columns named by dimension name, and the value column ``value``, in long format.

    The index labels are kept, as error messages use them to name rows.
    """
    renamed = {column: dim_name for dim_name, column in layout.dim_columns.items()}
    if layout.wide_dim is None:
        renamed[layout.value_column] = VALUE_COLUMN
        return df[list(renamed)].rename(columns=renamed)
    id_columns = list(layout.dim_columns.values())
    long_df = df[id_columns + list(layout.item_columns)].rename(columns=renamed)
    long_df = long_df.melt(
        id_vars=list(layout.dim_columns),
        value_vars=list(layout.item_columns),
        var_name=layout.wide_dim.name,
        value_name=VALUE_COLUMN,
        ignore_index=False,
    )
    long_df[layout.wide_dim.name] = long_df[layout.wide_dim.name].map(layout.item_columns)
    return long_df


def _layout_problem(
    df: pd.DataFrame,
    dims_without_column: list[Dimension],
    other_columns: list[Hashable],
    reason: str,
) -> str:
    """Explain that dimensions have no column, with exact hints on the probable cause."""
    dims_text = ", ".join(f"'{dim.name}' ({dim.letter})" for dim in dims_without_column)
    lines = [reason] if reason else []
    lines += [
        f"No column for dimension(s) {dims_text}. Columns: {list_values(df.columns)}.",
        "Dimension columns must be named by dimension name or letter.",
    ]
    hints = _column_hints(df, dims_without_column, other_columns)
    if len(dims_without_column) > 1 or not reason:
        for dim in dims_without_column:
            item_columns = [c for c in other_columns if as_item(c, dim) is not NOT_AN_ITEM]
            if item_columns:
                hints.append(
                    f"The names of columns {list_values(item_columns)} are items of "
                    f"'{dim.name}'. Wide format over '{dim.name}' needs a column for every "
                    "other dimension."
                )
    hints += _header_hints(df, dims_without_column)
    lines += [f"- {hint}" for hint in hints]
    return "\n".join(lines)


def _column_hints(
    df: pd.DataFrame, dims_without_column: list[Dimension], other_columns: list[Hashable]
) -> list[str]:
    """Hints for columns which probably belong to a dimension without column."""
    hints = []
    labels = {name: dim for dim in dims_without_column for name in (dim.name, dim.letter)}
    for column in other_columns:
        holds_items_of = [dim for dim in dims_without_column if _holds_items(df[column], dim)]
        if holds_items_of:
            dim = holds_items_of[0]
            if is_unnamed(column):
                hints.append(
                    f"Column {column!r} has no header, but holds items of '{dim.name}'. Name it "
                    f"'{dim.name}' or '{dim.letter}'."
                )
            elif as_item(column, dim) is not NOT_AN_ITEM:
                hints.append(
                    f"The first row seems to be data, not a header: column {column!r} is named by "
                    f"an item of '{dim.name}' and holds items of '{dim.name}'. The first row "
                    "must name the columns."
                )
            else:
                hints.append(
                    f"Column {column!r} holds items of '{dim.name}'. Name it '{dim.name}' or "
                    f"'{dim.letter}'."
                )
            continue
        meant = closest_name(column, labels)
        if meant is not None:
            hints.append(
                f"Column {column!r} is not named like a dimension. Did you mean '{meant}'?"
            )
    return hints


def _header_hints(df: pd.DataFrame, dims_without_column: list[Dimension]) -> list[str]:
    """Hints for data whose header is missing or not in the first row."""
    hints = []
    if list(df.columns) == list(range(len(df.columns))):
        hints.append(
            "The column names are 0, 1, 2, ...: the data have no header row, e.g. as read with "
            "header=None. The first row must name the columns."
        )
    elif sum(is_unnamed(c) for c in df.columns) >= len(df.columns) / 2:
        hints.append(
            "Most columns have no header. If the header is not in the first row, e.g. below a "
            "title, pass header=... or skiprows=... to pandas.read_csv or read_excel (also "
            "accepted as keyword arguments by the flodym readers)."
        )
    if not isinstance(df.index, pd.RangeIndex):
        for dim in dims_without_column:
            if _holds_items(df.index.to_series(), dim):
                hints.append(
                    f"The index holds items of '{dim.name}', but has no name. Name it "
                    f"'{dim.name}' or '{dim.letter}', or call reset_index() and name the column."
                )
    return hints


def _duplicate_hint(column: Hashable, dim: Dimension) -> str:
    """Hint at a column which pandas renamed because its header occurs twice."""
    match = RENAMED_DUPLICATE.match(str(column))
    if match is None or as_item(match.group(1), dim) is NOT_AN_ITEM:
        return ""
    return (
        f"Column {column!r}: the header {match.group(1)!r} seems to occur twice; pandas renamed "
        f"the second column to {column!r}."
    )


def _holds_items(entries: pd.Series, dim: Dimension) -> bool:
    """Whether all entries that are not empty are items of the dimension, and there are some."""
    unique = [entry for entry in entries.unique() if not _is_missing(entry)]
    return bool(unique) and all(as_item(entry, dim) is not NOT_AN_ITEM for entry in unique)


def _contains_numbers(entries: pd.Series) -> bool:
    """Whether any entry is a number or a string that can be read as a number."""
    return bool(pd.to_numeric(entries.dropna(), errors="coerce").notna().any())


def as_item(value: Any, dim: Dimension) -> Any:
    """The item of ``dim`` that ``value`` stands for, or ``NOT_AN_ITEM``.

    ``value`` is converted to the dtype of the dimension, if given. Otherwise, it is compared as
    text, as pandas reads all column names of CSV files as text, e.g. ``'2020'``.
    """
    if dim.dtype is not None:
        try:
            value = dim.dtype(value)
        except (ValueError, TypeError):
            return NOT_AN_ITEM
        return value if value in set(dim.items) else NOT_AN_ITEM
    items_by_text = {str(item): item for item in dim.items}
    return items_by_text.get(str(value), NOT_AN_ITEM)


def is_unnamed(column: Hashable) -> bool:
    """Whether pandas named the column because its header cell is empty."""
    return isinstance(column, str) and UNNAMED_COLUMN.match(column) is not None


def _type_hint(items: list[Any], dim: Dimension) -> str:
    """Hint at a missing dtype if items differ from those of the dimension only in their type."""
    dim_items_as_str = {str(item): item for item in dim.items}
    matching = [dim_items_as_str[str(item)] for item in items if str(item) in dim_items_as_str]
    if not matching or len(matching) < len(items) / 2:
        return ""
    given_type = type(items[0]).__name__
    dim_type = type(matching[0]).__name__
    if given_type == dim_type:
        return ""
    return (
        f"These items have type {given_type}, but the items of dimension '{dim.name}' have type "
        f"{dim_type}. Give the dimension the dtype {dim_type} (e.g. in its DimensionDefinition), "
        "so that the items in the data are converted."
    )


def _is_missing(value: Any) -> bool:
    """Whether a value is a missing value, e.g. NaN or None."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _combination(combination: Any) -> str:
    """Representation of an index combination, e.g. ``('EU', 2020)``."""
    if not isinstance(combination, tuple):
        combination = (combination,)
    return "(" + ", ".join(repr(to_builtin(item)) for item in combination) + ")"
