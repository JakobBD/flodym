"""Conversion of pandas DataFrames to values of FlodymArrays (private module)."""

import itertools
import logging
from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ._base_model import (
    MAX_LISTED_VALUES,
    FlodymBaseModel,
    FlodymDataError,
    closest_name,
    list_values,
    to_builtin,
)
from .dimensions import Dimension

if TYPE_CHECKING:
    from .flodym_arrays import FlodymArray

logger = logging.getLogger(__name__)

MAX_LISTED_COMBINATIONS = 10
"""Maximum number of missing index combinations listed in an error message."""

MIN_SHARE_OF_DIM_ITEMS = 0.5
"""A column is described as a possible dimension column in error messages if at least this
share of its items are items of the dimension."""

DEFAULT_VALUE_COLUMN = "value"
"""Name of the value column which allows further columns in long format to be ignored."""

NOT_AN_ITEM = object()
"""Returned by :py:meth:`DataFrameToFlodymDataConverter._as_item` for values which are not an
item of the dimension."""


class FlodymDataFormat(FlodymBaseModel):
    """Format of a DataFrame holding the values of a FlodymArray."""

    type: Literal["long", "wide"]
    """Long format: one column per dimension and one value column; wide format: the items of one
    dimension are spread over the value columns."""
    value_column: Hashable = DEFAULT_VALUE_COLUMN
    """Name of the value column (long format)."""
    columns_dim: str | None = None
    """Name of the dimension whose items are the value column names (wide format)."""


class DataFrameToFlodymDataConverter:
    """Converts a pandas DataFrame to a numpy array that can be used as values of a FlodymArray.

    Various DataFrame formats are possible.

    Usually not called by the user, but from within the FlodymArray from_df and
    set_values_from_df methods, where further documentation can be found.

    Problems with the data raise a :py:class:`flodym.FlodymDataError`. Where single entries are
    the problem, the message gives their labels in the index of the DataFrame. In case of
    errors, turning on debug logging might help to understand the process.
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
        self.allow_missing_values = allow_missing_values
        self.allow_extra_values = allow_extra_values
        self.strip_whitespace = strip_whitespace
        self.dim_columns: list[str] = []
        self.found_by: dict[str, str] = {}
        """How the column of each dimension was found: by "name", "letter" or "items"."""
        self.original_columns: dict[str, Hashable] = {}
        """Name of the column of each dimension in the given DataFrame, before renaming."""
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
        dims = self.flodym_array.dims
        dims_text = ", ".join(f"{dim.name} ({dim.letter})" for dim in dims) or "no dimensions"
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
        """Determine the DataFrame format and convert it to an array of the FlodymArray's shape.

        Returns:
            The values for the FlodymArray.
        """
        logger.debug(
            f"Start setting values for FlodymArray {self.flodym_array.name} with dimensions "
            f"{self.flodym_array.dims.names} from dataframe."
        )
        self._reset_non_default_index()
        if self.strip_whitespace:
            self._strip_whitespace_from_df()
        self._drop_empty_rows_and_columns()
        self._determine_format()
        self._df_to_long_format()
        self._check_missing_dim_columns()
        self._check_empty_dim_cells()
        self._convert_type()
        self._sort_columns()
        values = self._check_data_complete()
        return values

    @staticmethod
    def same_items(arr: Iterable[Any], dim: Dimension) -> bool:
        """Check if the items in ``arr`` are exactly the items of ``dim``.

        Items of ``arr`` are converted to the dimension dtype, if given. Order and repetitions are
        disregarded.
        """
        dim_items = set(dim.items)
        found_items = set()
        for item in arr:
            if dim.dtype is not None:
                try:
                    item = dim.dtype(item)
                except (ValueError, TypeError):
                    return False
            # Exit at the first non-matching item without scanning and converting all entries
            if item not in dim_items:
                return False
            found_items.add(item)
        return len(found_items) == len(dim_items)

    def _reset_non_default_index(self) -> None:
        if (
            isinstance(self.df.index, pd.MultiIndex)
            or self.df.index.name is not None
            or self.df.index.dtype != np.int64
            or self.df.index.min() >= 1700
            and self.df.index.max() <= 2300
        ):
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

        Such rows and columns occur e.g. in Excel sheets with formatted but empty cells. Columns
        without a name are named ``"Unnamed: <position>"`` by pandas.
        """
        empty_columns = [
            column
            for column in self.df.columns
            if isinstance(column, str)
            and column.startswith("Unnamed: ")
            and self.df[column].isna().all()
        ]
        if empty_columns:
            logger.debug(f"Dropping empty columns without name: {empty_columns}")
            self.df = self.df.drop(columns=empty_columns)
        self.df = self.df.dropna(how="all")
        if len(self.df.columns) == 0:
            raise FlodymDataError(
                "The DataFrame has no columns. Check that the file or sheet contains the data."
            )

    def _determine_format(self) -> None:
        self._get_dim_columns_by_name_or_letter()
        self._check_if_first_row_are_items()
        self._check_for_dim_columns_by_items()
        self._check_value_columns()

    def _get_dim_columns_by_name_or_letter(self) -> None:
        dims = self.flodym_array.dims
        self.original_dim_columns = [c for c in self.df.columns if c in dims]
        for c in self.df.columns:
            if c in dims.names:
                self.found_by[c] = "name"
            elif c in dims.letters:
                self.df.rename(columns={c: dims[c].name}, inplace=True)
                self.found_by[dims[c].name] = "letter"
                self.original_columns[dims[c].name] = c
                logger.debug(f"Renamed column {c} to dimension name {dims[c].name}")
        self.dim_columns = [c for c in self.df.columns if c in dims.names]
        logger.debug(f"Recognized index columns by name: {self.dim_columns}")

    def _check_if_first_row_are_items(self) -> None:
        """Restore the first row of the data if it was erroneously read as column names.

        If data without columns names was read, but the first row was assumed to be column names,
        the first row of the data frame might erroneously end up as column names.
        This method checks if that is the case, and if so, prepends a row based on column names.
        """
        column_name = self.df.columns[0]
        col_items = self.df[column_name].unique()
        extended_col_items = [column_name] + col_items.tolist()
        for dim in self.flodym_array.dims:
            if self.same_items(extended_col_items, dim):
                self._add_column_names_as_row(column_name, dim)
                return

    def _add_column_names_as_row(self, column_name: Hashable, dim: Dimension) -> None:
        if len(self.dim_columns) > 0:
            raise FlodymDataError(
                f"Ambiguity detected: column with first item {column_name!r} could be "
                f"dimension '{dim.name}' if the first row counts as an item, but columns "
                f"{self.original_dim_columns} are already recognized as dimensions by their "
                "name. Please add dimension names to your index columns, change the item names "
                "of the affected dimension, or use a different method to read data."
            )
        # prepend a row with the column names; its index label comes before all others
        first_label = self.df.index.min() - 1 if len(self.df) > 0 else 0
        header_row = pd.DataFrame([list(self.df.columns)], columns=self.df.columns)
        header_row.index = [first_label]
        self.df = pd.concat([header_row, self.df])
        self.df.columns = [f"column {i}" for i in range(len(self.df.columns))]

    def _check_for_dim_columns_by_items(self) -> None:
        for cn in self.df.columns:
            if cn in self.dim_columns:
                continue
            found = self._check_if_dim_column_by_items(cn)
            if not found:
                logger.debug(
                    f"Could not find dimension with same items as column {cn}. "
                    "Assuming this is the first value column; Won't look further."
                )
                return

    def _check_if_dim_column_by_items(self, column_name: Hashable) -> bool:
        logger.debug(f"Checking if {column_name} is a dimension by comparing items with dim items")
        col_items = self.df[column_name].unique()
        for dim in self.flodym_array.dims:
            if dim.name in self.dim_columns:
                continue
            if self.same_items(col_items, dim):
                logger.debug(f"{column_name} is dimension {dim.name}.")
                self.df.rename(columns={column_name: dim.name}, inplace=True)
                self.dim_columns.append(dim.name)
                self.found_by[dim.name] = "items"
                self.original_columns[dim.name] = column_name
                return True
        return False

    def _check_value_columns(self) -> None:
        value_columns = [c for c in self.df.columns if c not in self.dim_columns]
        logger.debug(f"Assumed value columns: {value_columns}")
        if not value_columns:
            raise FlodymDataError(
                f"No value column found: all columns ({list_values(self.df.columns)}) were found "
                "to be dimension columns."
            )
        if not self._check_if_value_columns_match_dim_items(value_columns):
            self._check_if_valid_long_format(value_columns)

    def _check_if_value_columns_match_dim_items(self, value_columns: list[Hashable]) -> bool:
        """Check for wide format: the first value columns are named by the items of a dimension.

        Further columns to their right are dropped if they contain no numbers, e.g. units or
        comments. If extra values are allowed, they are dropped in any case.

        Returns:
            Whether the data are in wide format.
        """
        logger.debug("Trying to match set of value column names with items of dimension.")
        for dim in self.flodym_array.dims:
            if dim.name in self.dim_columns:
                continue
            n_item_columns = 0
            for column in value_columns:
                if self._as_item(column, dim) is NOT_AN_ITEM:
                    break
                n_item_columns += 1
            item_columns = value_columns[:n_item_columns]
            if not item_columns or not self.same_items(item_columns, dim):
                continue
            logger.debug(f"Value columns match dimension items of {dim.name}.")
            self._drop_columns_right_of_values(value_columns[n_item_columns:], dim)
            self.format = FlodymDataFormat(type="wide", columns_dim=dim.name)
            if dim.dtype is not None:
                self.df.rename(columns={c: dim.dtype(c) for c in item_columns}, inplace=True)
            return True
        return False

    def _drop_columns_right_of_values(self, columns: list[Hashable], dim: Dimension) -> None:
        """Drop columns right of the value columns of the wide format, if they hold no values.

        Raises:
            FlodymDataError: If one of the columns contains numbers, and extra values are not
                allowed.
        """
        if not columns:
            return
        if not self.allow_extra_values:
            numeric_columns = [c for c in columns if self._contains_numbers(c)]
            if numeric_columns:
                raise FlodymDataError(
                    f"The value columns are named by the items of dimension '{dim.name}', but "
                    "further columns right of them contain numbers: "
                    f"{list_values(numeric_columns)}.\n"
                    f"If these are values for items that are not in dimension '{dim.name}', set "
                    "allow_extra_values=True to ignore them. Otherwise, remove these columns. "
                    "Columns right of the value columns are ignored only if they contain no "
                    "numbers, e.g. units or comments."
                )
        logger.info(
            f"Ignoring columns right of the value columns of '{self.flodym_array.name}': "
            f"{list_values(columns)}"
        )
        self.df = self.df.drop(columns=columns)

    def _contains_numbers(self, column: Hashable) -> bool:
        """Whether any entry of a column is a number or a string that can be read as a number."""
        entries = self.df[column].dropna()
        return bool(pd.to_numeric(entries, errors="coerce").notna().any())

    def _check_if_valid_long_format(self, value_columns: list[Hashable]) -> None:
        """Check for long format: one value column, or a column named "value".

        Further columns are dropped if the value column is named "value" and all dimension
        columns were found by their name or letter, as then no column can be mistaken for another.

        Raises:
            FlodymDataError: If the value column cannot be determined.
        """
        logger.debug(
            "Could not find dimension with same item set as value column names. Assuming long "
            "format, i.e. one value column."
        )
        if len(value_columns) == 1:
            self.format = FlodymDataFormat(type="long", value_column=value_columns[0])
            logger.debug(f"Value column name is {value_columns[0]}.")
            return
        if DEFAULT_VALUE_COLUMN in value_columns and self._dims_found_by_label():
            ignored = [c for c in value_columns if c != DEFAULT_VALUE_COLUMN]
            logger.info(
                f"Ignoring columns of '{self.flodym_array.name}' which are neither dimension "
                f"columns nor the value column '{DEFAULT_VALUE_COLUMN}': {list_values(ignored)}"
            )
            self.df = self.df.drop(columns=ignored)
            self.format = FlodymDataFormat(type="long", value_column=DEFAULT_VALUE_COLUMN)
            return
        raise FlodymDataError(self._format_problem(value_columns))

    def _dims_found_by_label(self) -> bool:
        """Whether all dimensions with more than one item were found by their name or letter."""
        return all(
            self.found_by.get(dim.name) in ("name", "letter")
            for dim in self.flodym_array.dims
            if dim.len > 1
        )

    def _format_problem(self, value_columns: list[Hashable]) -> str:
        """Explain why the format could not be determined, with the closest matches found."""
        lines = ["Could not determine which column(s) hold the values."]
        if self.dim_columns:
            found = [
                f"{self.original_columns.get(c, c)!r} for '{c}' (by {self.found_by.get(c, 'items')})"
                for c in self.dim_columns
            ]
            lines.append(f"Columns found for dimensions: {', '.join(found)}.")
        else:
            lines.append("No column was found for any dimension.")
        lines.append(f"Other columns: {list_values(value_columns)}.")
        missing_dims = [
            dim
            for dim in self.flodym_array.dims
            if dim.name not in self.dim_columns and dim.len > 1
        ]
        explained_dims: set[str] = set()
        wide_dims: set[str] = set()
        for dim in missing_dims:
            wide_hint = self._wide_format_hint(value_columns, dim)
            if wide_hint:
                lines.append(wide_hint)
                explained_dims.add(dim.name)
                wide_dims.add(dim.name)
        for column in value_columns:
            column_hint = self._column_hint(column, missing_dims)
            if column_hint is not None:
                lines.append(column_hint[1])
                explained_dims.add(column_hint[0])
        for dim in missing_dims:
            if dim.name not in explained_dims:
                lines.append(
                    f"- No column matches dimension '{dim.name}' ({dim.letter}), with items "
                    f"{list_values(dim.items)}."
                )
        lines.append(
            "Expected are either one column for each dimension and one value column (long "
            "format), or one column for each dimension but one, and one value column for each "
            "item of that dimension (wide format)."
        )
        if DEFAULT_VALUE_COLUMN in value_columns:
            lines.append(
                f"Further columns are ignored if the value column is named "
                f"'{DEFAULT_VALUE_COLUMN}' and all dimension columns are named by dimension name "
                "or letter."
            )
        lines += self._naming_hint(wide_dims)
        return "\n".join(lines)

    def _wide_format_hint(self, value_columns: list[Hashable], dim: Dimension) -> str:
        """Describe how the column names differ from the items of a dimension, if they are close."""
        item_columns = [c for c in value_columns if self._as_item(c, dim) is not NOT_AN_ITEM]
        if not item_columns or len(item_columns) < len(value_columns) / 2:
            return ""
        lines = [
            (
                f"- The column names {list_values(item_columns)} are items of dimension "
                f"'{dim.name}', as in wide format, but:"
            ),
        ]
        items_with_column = {self._as_item(c, dim) for c in item_columns}
        missing_items = [item for item in dim.items if item not in items_with_column]
        if missing_items:
            lines.append(f"  - Items without column: {list_values(missing_items)}.")
        first = value_columns.index(item_columns[0])
        last = value_columns.index(item_columns[-1])
        before = value_columns[:first]
        if before:
            lines.append(
                f"  - Further columns left of them: {list_values(before)}. Further columns are "
                "only allowed right of the value columns."
            )
        between = [c for c in value_columns[first:last] if c not in item_columns]
        if between:
            lines.append(f"  - Further columns between them: {list_values(between)}.")
        numeric_after = [c for c in value_columns[last + 1 :] if self._contains_numbers(c)]
        if numeric_after and not self.allow_extra_values:
            lines.append(
                f"  - Further columns right of them contain numbers: {list_values(numeric_after)}. "
                "They are only ignored if they contain no numbers, e.g. units or comments, or "
                "if allow_extra_values=True."
            )
        return "\n".join(lines) if len(lines) > 1 else ""

    def _column_hint(self, column: Hashable, dims: list[Dimension]) -> tuple[str, str] | None:
        """Describe how the items of a column differ from those of the closest dimension.

        Returns:
            The name of the closest dimension and the description, or None if no dimension is
            close.
        """
        dims_by_label = {dim.name: dim for dim in dims} | {dim.letter: dim for dim in dims}
        meant_label = closest_name(column, dims_by_label)
        if meant_label is not None:
            return dims_by_label[meant_label].name, (
                f"- Column {column!r} is not named like a dimension. Did you mean "
                f"'{meant_label}'? Dimension columns must be named exactly by dimension name or "
                "letter."
            )
        column_items = [item for item in self.df[column].unique() if not _is_missing(item)]
        if not column_items or not dims:
            return None

        def share_in_dim(dim: Dimension) -> float:
            dim_items = {str(item).strip() for item in dim.items}
            return sum(str(item).strip() in dim_items for item in column_items) / len(column_items)

        closest = max(dims, key=share_in_dim)
        share = share_in_dim(closest)
        if share < MIN_SHARE_OF_DIM_ITEMS:
            return None
        if self.same_items(column_items, closest):
            return closest.name, (
                f"- Column '{column}' has the items of dimension '{closest.name}', but is right of "
                "a column which is not a dimension column. Columns without dimension name or "
                "letter are found by their items only from the left, up to the first column which "
                "is not a dimension column."
            )
        lines = [
            (
                f"- Column '{column}' could be dimension '{closest.name}' ({share:.0%} of its "
                "items are items of the dimension), but:"
            ),
        ]
        extra = [item for item in column_items if self._as_item(item, closest) is NOT_AN_ITEM]
        if extra:
            lines.append(
                "  - Items not in the dimension: "
                + self._entries_with_index(column, extra, suggestions=closest.items)
                + "."
            )
            type_hint = _type_hint(extra, closest)
            if type_hint:
                lines.append(f"  - {type_hint}")
        found = {self._as_item(item, closest) for item in column_items}
        missing = [item for item in closest.items if item not in found]
        if missing:
            lines.append(f"  - Dimension items not in the column: {list_values(missing)}.")
        return closest.name, "\n".join(lines)

    def _naming_hint(self, wide_dims: set[str]) -> list[str]:
        """Recommend naming the dimension columns, if they were not all found by name or letter.

        Args:
            wide_dims: Dimensions which are probably given by the value column names.
        """
        dims = [
            dim
            for dim in self.flodym_array.dims
            if dim.len > 1
            and dim.name not in wide_dims
            and self.found_by.get(dim.name) not in ("name", "letter")
        ]
        if not dims:
            return []
        names = ", ".join(f"'{dim.name}' or '{dim.letter}'" for dim in dims)
        return [
            (
                f"Tip: name the dimension columns by dimension name or letter ({names}). Then "
                "they are found regardless of their items, and error messages are more precise."
            ),
        ]

    def _df_to_long_format(self) -> None:
        if self.format.type != "wide":
            return
        logger.debug("Converting wide format to long format.")
        value_cols = self.flodym_array.dims[self.format.columns_dim].items
        self.df = self.df.melt(
            id_vars=[c for c in self.df.columns if c not in value_cols],
            value_vars=value_cols,
            var_name=self.format.columns_dim,
            value_name=self.format.value_column,
            ignore_index=False,  # keep the index labels to name rows in error messages
        )
        self.dim_columns.append(self.format.columns_dim)
        self.format = FlodymDataFormat(type="long", value_column=self.format.value_column)

    def _check_missing_dim_columns(self) -> None:
        missing_dim_columns = [
            dim.name for dim in self.flodym_array.dims if dim.name not in self.dim_columns
        ]
        for c in missing_dim_columns:
            dim = self.flodym_array.dims[c]
            if dim.len == 1:
                self.df[c] = dim.items[0]
                self.dim_columns.append(c)
            else:
                raise FlodymDataError(
                    f"No column found for dimension '{dim.name}' ({dim.letter}), which has "
                    f"{dim.len} items: {list_values(dim.items)}. Columns of the DataFrame: "
                    f"{list_values(self.df.columns)}. Name the column of this dimension "
                    f"'{dim.name}' or '{dim.letter}'."
                )

    def _check_empty_dim_cells(self) -> None:
        """Check for empty cells in dimension columns; drop their rows if extra values are allowed.

        Raises:
            FlodymDataError: If there are empty cells and extra values are not allowed.
        """
        for dim in self.flodym_array.dims:
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
        for dim in self.flodym_array.dims:
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
        value_column = self.format.value_column
        values = pd.to_numeric(self.df[value_column], errors="coerce")
        not_numbers = values.isna() & self.df[value_column].notna()
        if not_numbers.any():
            entries = self.df.loc[not_numbers, value_column].unique().tolist()
            raise FlodymDataError(
                f"Value column '{value_column}' has entries that are not numbers: "
                f"{self._entries_with_index(value_column, entries)}."
            )
        self.df[value_column] = values.astype(np.float64)

    def _sort_columns(self) -> None:
        """Sort the data frame columns according to the order of the FlodymArray dimensions."""
        self.df = self.df[list(self.flodym_array.dims.names) + [self.format.value_column]]

    def _check_data_complete(self) -> np.ndarray:
        # Special handling of the scalar case: If the FlodymArray has no dimensions, we usually
        # expect a single value in the df.
        if self.flodym_array.dims.ndim == 0:
            if len(self.df) == 0:
                if self.allow_missing_values:
                    return np.array(0.0)
                raise FlodymDataError(
                    "FlodymArray has no dimensions, but the DataFrame has no rows. Expected "
                    "exactly one row."
                )
            if len(self.df) > 1:
                raise FlodymDataError(
                    f"FlodymArray has no dimensions, but the DataFrame has {len(self.df)} rows. "
                    "Expected exactly one row."
                )
            return np.array(self.df[self.format.value_column].values[0])

        self._check_no_duplicates()

        # remove rows with extra values or throw error
        if self.allow_extra_values:
            for dim in self.flodym_array.dims:
                self.df = self.df[self.df[dim.name].isin(dim.items)]
        else:
            for dim in self.flodym_array.dims:
                self._check_no_extra_items(dim)

        if self.allow_missing_values:
            self.df[self.format.value_column] = self.df[self.format.value_column].fillna(0)
        else:
            # Duplicates throw an error, and extra values are removed above,
            # so unequal lengths indicate missing values
            if len(self.df) != self.flodym_array.size:
                self._raise_missing_values()
            empty = self.df[self.format.value_column].isna()
            if empty.any():
                dim_names = list(self.flodym_array.dims.names)
                combinations = self.df.loc[empty, dim_names].itertuples(index=True, name=None)
                listed = ", ".join(
                    f"{_combination(tuple(combination[1:]))} (index {combination[0]!r})"
                    for combination in itertools.islice(combinations, MAX_LISTED_VALUES)
                )
                n_empty = int(empty.sum())
                more = f", ... ({n_empty} in total)" if n_empty > MAX_LISTED_VALUES else ""
                raise FlodymDataError(
                    f"Values are empty for these index combinations ({', '.join(dim_names)}): "
                    f"{listed}{more}.\nFill them, or set allow_missing_values=True to set them to "
                    "zero."
                )

        # for performance and memory reasons, we use numpy operations to convert the data to the
        # numpy array
        # replace non-value items in df with their index in dims.items
        for dim in self.flodym_array.dims:
            self.df[dim.name] = self.df[dim.name].map({item: i for i, item in enumerate(dim.items)})
        # convert df to numpy index array: Take all non-value columns
        fill_indices = self.df.values[:, :-1].T.astype(np.intp)
        fill_values = self.df[self.format.value_column].values
        # this is what ends up in the parameter; initialize with zeros
        values = np.zeros(self.flodym_array.dims.shape)
        values[tuple(fill_indices)] = fill_values
        return values

    def _check_no_duplicates(self) -> None:
        """Raise an error if index combinations occur more than once."""
        indices = self.df[list(self.flodym_array.dims.names)]
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
        indices = self.df[list(self.flodym_array.dims.names)]
        actual_index_tuples = set(indices.itertuples(index=False, name=None))
        missing = [
            combination
            for combination in itertools.product(*[dim.items for dim in self.flodym_array.dims])
            if combination not in actual_index_tuples
        ]
        listed = ", ".join(_combination(c) for c in missing[:MAX_LISTED_COMBINATIONS])
        more = ", ..." if len(missing) > MAX_LISTED_COMBINATIONS else ""
        raise FlodymDataError(
            f"Values are missing for {len(missing)} of {self.flodym_array.size} index "
            f"combinations ({', '.join(self.flodym_array.dims.names)}): {listed}{more}\n"
            "Add them, or set allow_missing_values=True to set them to zero."
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

    @staticmethod
    def _as_item(value: Any, dim: Dimension) -> Any:
        """The item of ``dim`` that ``value`` stands for, or ``NOT_AN_ITEM``.

        ``value`` is converted to the dtype of the dimension, if given.
        """
        if dim.dtype is not None:
            try:
                value = dim.dtype(value)
            except (ValueError, TypeError):
                return NOT_AN_ITEM
        try:
            return value if value in dim.items else NOT_AN_ITEM
        except TypeError:  # unhashable or incomparable value
            return NOT_AN_ITEM


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
