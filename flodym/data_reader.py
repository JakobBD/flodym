"""Home to some data readers."""

import os
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from os import PathLike
from typing import Any, TypeVar

import pandas as pd

from ._base_model import FlodymDataError, did_you_mean, list_values
from .dimensions import Dimension, DimensionSet
from .flodym_arrays import Parameter
from .mfa_definition import DimensionDefinition, ParameterDefinition, _dimension_dependency_order

ReadResult = TypeVar("ReadResult")


def _lookup(mapping: Mapping[str, Any], key: str, argument_name: str, what: str) -> Any:
    """Get the entry of a reader argument such as ``parameter_files`` for a parameter or dimension.

    Raises:
        FlodymDataError: If there is no entry for ``key``, with the given keys and a suggestion.
    """
    if key in mapping:
        return mapping[key]
    raise FlodymDataError(
        f"No entry for {what} '{key}' in {argument_name}.{did_you_mean(key, mapping)} "
        f"Given entries: {list_values(mapping) or 'none'}."
    )


def _read_with_context(
    read: Callable[[], ReadResult],
    subject: str,
    path: str | PathLike[str],
    argument_name: str,
    sheet_name: str | int | None = None,
    index_note: str = "",
) -> ReadResult:
    """Run ``read``, and name the file and what is read from it in errors.

    Args:
        read: Function that reads the file and creates the flodym object.
        subject: What is read, e.g. ``"parameter 'p'"``.
        path: The file that is read.
        argument_name: Name of the reader argument which gives the file, e.g.
            ``"parameter_files"``.
        sheet_name: The sheet that is read, for Excel files.
        index_note: Explanation of the index of the read DataFrame, which error messages use
            to name rows, e.g. ``" (index = row number in the file)"``.

    Raises:
        FileNotFoundError: If the file does not exist, with the absolute path.
        FlodymDataError: If the data are invalid, or pandas cannot read them.
    """
    sheet = f", sheet {sheet_name!r}" if isinstance(sheet_name, str) else ""
    context = f"Could not read {subject} from file '{path}'{sheet}{index_note}:"
    try:
        return read()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"The file for {subject} given in {argument_name} does not exist: '{path}'. "
            f"Relative paths are relative to the working directory, '{os.getcwd()}'."
        ) from error
    except FlodymDataError as error:
        error._add_context(context, subject)
        raise
    except ValueError as error:
        # e.g. errors of pandas when parsing the file, or invalid arguments of flodym classes
        message = str(error) if type(error) is ValueError else f"{type(error).__name__}: {error}"
        wrapped = FlodymDataError(message)
        wrapped._add_context(context, subject)
        raise wrapped from error


def _add_reading_context(error: BaseException, context: str, subject: str) -> None:
    """Name what was read when an error occurred in a method of a reader.

    For a :py:class:`flodym.FlodymDataError`, the context is prepended to the message, unless the
    message already names what was read. Other errors are not changed, as their type may matter
    to the caller, but get a note (Python 3.11 and later), which is shown below the traceback.
    """
    if isinstance(error, FlodymDataError):
        if error._subject != subject:
            error._add_context(context, subject)
    elif sys.version_info >= (3, 11):
        error.add_note(context.rstrip(":"))


def _keeps_file_rows(read_kwargs: dict[str, Any]) -> bool:
    """Whether data read with these keyword arguments start in row 2 of the file, after a header.

    This holds for pandas' default header, index and skipped rows. Then the index of the read
    DataFrame can be set to the row numbers in the file, which error messages use to name rows.
    """
    return not any(key in read_kwargs for key in ("header", "skiprows", "index_col", "nrows"))


def _set_index_to_file_rows(data: pd.DataFrame) -> None:
    """Set the index of data read with header in row 1 to the row numbers in the file."""
    data.index = pd.RangeIndex(2, len(data) + 2)


def _is_allowed_for(allowed: bool | list[str], parameter_name: str) -> bool:
    """Resolve an ``allow_*`` setting of a parameter reader for one parameter.

    Args:
        allowed: Either a flag applying to all parameters, or the names of the parameters for
            which the setting is enabled.
        parameter_name: Name of the parameter that is read.

    Returns:
        Whether the setting is enabled for the parameter.
    """
    if isinstance(allowed, bool):
        return allowed
    return parameter_name in allowed


class DataReader(ABC):
    """Template for creating a data reader.

    Shows the required methods and data formats needed for use in the MFASystem model.
    """

    def read_dimensions(self, dimension_definitions: list[DimensionDefinition]) -> DimensionSet:
        """Method to read data for multiple dimensions, by looping over `read_dimension`.

        Sub-dimensions (see :py:attr:`flodym.DimensionDefinition.parent`) are linked to their
        parent. Dimensions with :py:attr:`flodym.DimensionDefinition.splitting` are not read, but
        created from their parent with :py:meth:`flodym.Dimension.split`. Likewise, dimensions with
        :py:attr:`flodym.DimensionDefinition.combine_splits` are created with
        :py:meth:`flodym.Dimension.combine_splits`.

        Args:
            dimension_definitions: Definitions of the dimensions to read.

        Returns:
            The dimensions, in the order of the definitions.
        """
        dimensions: dict[str, Dimension] = {}
        for definition in _dimension_dependency_order(dimension_definitions):
            try:
                dimensions[definition.letter] = self._create_dimension(definition, dimensions)
            except Exception as error:
                _add_reading_context(
                    error,
                    f"While reading dimension '{definition.name}' ({definition.letter}) with "
                    f"{type(self).__name__}:",
                    f"dimension '{definition.name}'",
                )
                raise
        return DimensionSet(dim_list=[dimensions[d.letter] for d in dimension_definitions])

    def _create_dimension(
        self, definition: DimensionDefinition, dimensions: dict[str, Dimension]
    ) -> Dimension:
        """Read a dimension, or create it from the dimensions it depends on.

        Args:
            definition: Definition of the dimension.
            dimensions: Already created dimensions by letter, including those the definition
                depends on.
        """
        if definition.splitting is not None:
            return _split_dimension(definition, dimensions)
        if definition.combine_splits is not None:
            upper, *lower = (dimensions[letter] for letter in definition.combine_splits)
            combined = Dimension.combine_splits(
                name=definition.name, letter=definition.letter, upper=upper, lower=lower
            )
            return Dimension.model_validate({**dict(combined), "dtype": definition.dtype})
        dimension = self.read_dimension(definition)
        if definition.parent is not None:
            dimension = _link_parent(dimension, definition, dimensions[definition.parent])
        return dimension

    @abstractmethod
    def read_dimension(self, definition: DimensionDefinition) -> Dimension:
        """Required method to read data for a single dimension, given its definition."""

    @abstractmethod
    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        """Required method to read data for a particular parameter."""

    def read_parameters(
        self, parameter_definitions: list[ParameterDefinition], dims: DimensionSet
    ) -> dict[str, Parameter]:
        """Method to read data for a list of parameters, by looping over `read_parameter_values`.

        The ``extensive`` flag of a definition is set on the parameter if it is not None.

        Args:
            parameter_definitions: Definitions of the parameters to read.
            dims: Dimensions of the MFA system; each parameter receives the subset given by the
                ``dim_letters`` of its definition.

        Returns:
            The parameters, keyed by parameter name.
        """
        parameters = {}
        for parameter_definition in parameter_definitions:
            dim_subset = dims.get_subset(parameter_definition.dim_letters)
            try:
                parameter = self.read_parameter_values(
                    parameter_name=parameter_definition.name,
                    dims=dim_subset,
                )
            except Exception as error:
                dims_text = ", ".join(f"{dim.name} ({dim.letter})" for dim in dim_subset)
                _add_reading_context(
                    error,
                    f"While reading parameter '{parameter_definition.name}' with dimensions "
                    f"{dims_text or 'none'} with {type(self).__name__}.read_parameter_values:",
                    f"parameter '{parameter_definition.name}'",
                )
                raise
            if parameter_definition.extensive is not None:
                parameter.extensive = parameter_definition.extensive
            parameters[parameter_definition.name] = parameter
        return parameters


class DimensionReader(ABC):
    """Template for creating a dimension reader.

    Shows the required methods and data formats needed.
    """

    read_dimensions = DataReader.read_dimensions
    _create_dimension = DataReader._create_dimension

    @abstractmethod
    def read_dimension(self, definition: DimensionDefinition) -> Dimension:
        """Required method to read data for a single dimension, given its definition."""


class CSVDimensionReader(DimensionReader):
    """Read dimensions from a CSV file.

    Expects a single row or single column csv file with no header, containing the dimension items.
    For a dimension with a :py:attr:`flodym.DimensionDefinition.parent`, the file has two columns:
    the items, and the parent item of each item.

    Args:
        dimension_files: {dimension_name: file_path, ...}
        read_csv_kwargs: Additional keyword arguments passed to pandas.read_csv.
            The default is {"header": None}. Not encouraged to use, since it may not lead to the
            intended DataFrame format. Sticking to recommended csv file format is preferred.
    """

    def __init__(
        self,
        dimension_files: dict[str, str | PathLike[str]],
        **read_csv_kwargs: Any,
    ) -> None:
        self.dimension_files = dimension_files
        self.read_csv_kwargs = read_csv_kwargs

    def read_dimension(self, definition: DimensionDefinition) -> Dimension:
        """Read the items of a dimension from the CSV file given for the dimension name.

        Raises:
            FlodymDataError: If no file is given for the dimension, or its data are invalid.
            FileNotFoundError: If the file does not exist.
        """
        path = _lookup(self.dimension_files, definition.name, "dimension_files", "dimension")
        if "header" not in self.read_csv_kwargs:
            self.read_csv_kwargs["header"] = None

        def read() -> Dimension:
            df = pd.read_csv(path, **self.read_csv_kwargs)
            return Dimension.from_df(df, definition)

        return _read_with_context(read, f"dimension '{definition.name}'", path, "dimension_files")


class ExcelDimensionReader(DimensionReader):
    """Read dimensions from Excel file(s).

    Expects a single row or single column excel sheet with no header, containing the dimension
    items. For a dimension with a :py:attr:`flodym.DimensionDefinition.parent`, the sheet has two
    columns: the items, and the parent item of each item.

    Args:
        dimension_files: {dimension_name: file_path, ...}
        dimension_sheets: {dimension_name: sheet_name, ...}. If None, the first sheet is used.
        read_excel_kwargs: Additional keyword arguments passed to pandas.read_excel.
            The default is {"header": None}. Not encouraged to use, since it may not lead to the
            intended DataFrame format. Sticking to recommended excel file format is preferred.
    """

    def __init__(
        self,
        dimension_files: dict[str, str | PathLike[str]],
        dimension_sheets: dict[str, str] | None = None,
        **read_excel_kwargs: Any,
    ) -> None:
        self.dimension_files = dimension_files
        self.dimension_sheets = dimension_sheets
        self.read_excel_kwargs = read_excel_kwargs

    def read_dimension(self, definition: DimensionDefinition) -> Dimension:
        """Read the items of a dimension from the Excel file and sheet given for its name.

        Raises:
            FlodymDataError: If no file or sheet is given for the dimension, or its data are
                invalid.
            FileNotFoundError: If the file does not exist.
        """
        path = _lookup(self.dimension_files, definition.name, "dimension_files", "dimension")
        if self.dimension_sheets is None:
            sheet_name = 0  # first sheet; None would make pandas read all sheets into a dict
        else:
            sheet_name = _lookup(
                self.dimension_sheets, definition.name, "dimension_sheets", "dimension"
            )
        # default for header is None
        if "header" not in self.read_excel_kwargs:
            self.read_excel_kwargs["header"] = None

        def read() -> Dimension:
            df = pd.read_excel(path, sheet_name=sheet_name, **self.read_excel_kwargs)
            return Dimension.from_df(df, definition)

        return _read_with_context(
            read, f"dimension '{definition.name}'", path, "dimension_files", sheet_name
        )


class ParameterReader(ABC):
    """Template for creating a parameter reader.

    Shows the required methods and data formats needed.
    """

    @abstractmethod
    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        """Required method to read data for a particular parameter."""

    read_parameters = DataReader.read_parameters


class CSVParameterReader(ParameterReader):
    """Reads a csv file to a pandas data frame and calls :py:meth:`flodym.Parameter.from_df`.

    Expects comma separation and no header, apart from optional column names.
    For further detail on expected format, see :py:meth:`flodym.FlodymArray.from_df`.

    Args:
        parameter_files: Mapping of parameter names to file paths.
            Format: {parameter_name: file_path, ...}
        allow_missing_values: Whether to allow missing values in the DataFrame.
            This includes both missing rows, and NaN values in the value column.
            If a list of parameter names is given, missing values are only allowed for these
            parameters. Defaults to False.
        allow_extra_values: Whether to allow extra rows in the DataFrame,
            i.e. rows with index items not present in the FlodymArray dimension items.
            If a list of parameter names is given, extra rows are only allowed for these
            parameters. Defaults to False.
        read_csv_kwargs: Additional keyword arguments passed to pandas.read_csv.
            Not encouraged to use, since it may not lead to the intended DataFrame format.
            Sticking to recommended csv file format is preferred.
    """

    def __init__(
        self,
        parameter_files: dict[str, str | PathLike[str]],
        allow_missing_values: bool | list[str] = False,
        allow_extra_values: bool | list[str] = False,
        **read_csv_kwargs: Any,
    ) -> None:
        self.parameter_files = parameter_files
        self.allow_missing_values = allow_missing_values
        self.allow_extra_values = allow_extra_values
        self.read_csv_kwargs = read_csv_kwargs

    @property
    def parameter_filenames(self) -> dict[str, str | PathLike[str]]:
        """Deprecated. Use :py:attr:`parameter_files` instead."""
        warnings.warn(
            "parameter_filenames is deprecated since v2.0.0. It will be removed in v3.0.0. "
            "Use parameter_files instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.parameter_files

    @parameter_filenames.setter
    def parameter_filenames(self, value: dict[str, str | PathLike[str]]) -> None:
        warnings.warn(
            "parameter_filenames is deprecated since v2.0.0. It will be removed in v3.0.0. "
            "Use parameter_files instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.parameter_files = value

    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        """Read the values of a parameter from the CSV file given for the parameter name.

        Raises:
            FlodymDataError: If no file is given for the parameter, or its data are invalid.
            FileNotFoundError: If the file does not exist.
        """
        path = _lookup(self.parameter_files, parameter_name, "parameter_files", "parameter")
        keeps_file_rows = _keeps_file_rows(self.read_csv_kwargs)

        def read() -> Parameter:
            data = pd.read_csv(path, **self.read_csv_kwargs)
            if keeps_file_rows:
                _set_index_to_file_rows(data)
            return Parameter.from_df(
                dims=dims,
                name=parameter_name,
                df=data,
                allow_missing_values=_is_allowed_for(self.allow_missing_values, parameter_name),
                allow_extra_values=_is_allowed_for(self.allow_extra_values, parameter_name),
            )

        index_note = (
            " (index = line number in the file, if it has no empty lines)"
            if keeps_file_rows
            else ""
        )
        return _read_with_context(
            read, f"parameter '{parameter_name}'", path, "parameter_files", index_note=index_note
        )


class ExcelParameterReader(ParameterReader):
    """Reads an excel file to a pandas data frame and calls :py:meth:`flodym.Parameter.from_df`.

    Expects contiguous data starting in the upper left cell A1.
    For further detail on expected format, see :py:meth:`flodym.FlodymArray.from_df`.

    Args:
        parameter_files: Mapping of parameter names to file paths.
            Can be the same file for multiple parameters if the sheets are different.
            Format: {parameter_name: file_path, ...}
        parameter_sheets: Mapping of parameter names to sheet names in the excel file.
            If None, the first sheet is used.
            Format: {parameter_name: sheet_name, ...}
            Defaults to None.
        allow_missing_values: Whether to allow missing values in the DataFrame.
            This includes both missing rows, and NaN values in the value column.
            If a list of parameter names is given, missing values are only allowed for these
            parameters. Defaults to False.
        allow_extra_values: Whether to allow extra rows in the DataFrame,
            i.e. rows with index items not present in the FlodymArray dimension items.
            If a list of parameter names is given, extra rows are only allowed for these
            parameters. Defaults to False.
        read_excel_kwargs: Additional keyword arguments passed to pandas.read_excel.
            Not encouraged to use, since it may not lead to the intended DataFrame format.
            Sticking to recommended excel file format is preferred
    """

    def __init__(
        self,
        parameter_files: dict[str, str | PathLike[str]],
        parameter_sheets: dict[str, str] | None = None,
        allow_missing_values: bool | list[str] = False,
        allow_extra_values: bool | list[str] = False,
        **read_excel_kwargs: Any,
    ) -> None:
        self.parameter_files = parameter_files
        self.parameter_sheets = parameter_sheets
        self.allow_missing_values = allow_missing_values
        self.allow_extra_values = allow_extra_values
        self.read_excel_kwargs = read_excel_kwargs

    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        """Read the values of a parameter from the Excel file and sheet given for its name.

        Raises:
            FlodymDataError: If no file or sheet is given for the parameter, or its data are
                invalid.
            FileNotFoundError: If the file does not exist.
        """
        path = _lookup(self.parameter_files, parameter_name, "parameter_files", "parameter")
        if self.parameter_sheets is None:
            sheet_name = 0  # first sheet; None would make pandas read all sheets into a dict
        else:
            sheet_name = _lookup(
                self.parameter_sheets, parameter_name, "parameter_sheets", "parameter"
            )
        keeps_file_rows = _keeps_file_rows(self.read_excel_kwargs)

        def read() -> Parameter:
            data = pd.read_excel(path, sheet_name=sheet_name, **self.read_excel_kwargs)
            if keeps_file_rows:
                _set_index_to_file_rows(data)
            return Parameter.from_df(
                dims=dims,
                name=parameter_name,
                df=data,
                allow_missing_values=_is_allowed_for(self.allow_missing_values, parameter_name),
                allow_extra_values=_is_allowed_for(self.allow_extra_values, parameter_name),
            )

        index_note = " (index = row number in the file)" if keeps_file_rows else ""
        return _read_with_context(
            read,
            f"parameter '{parameter_name}'",
            path,
            "parameter_files",
            sheet_name,
            index_note=index_note,
        )


class CompoundDataReader(DataReader):
    """Combines a DimensionReader and a ParameterReader to create a DataReader.

    The resulting reader reads both dimensions and parameters.

    Args:
        dimension_reader: Reader used for all dimensions.
        parameter_reader: Reader used for all parameters.
    """

    def __init__(
        self,
        dimension_reader: DimensionReader,
        parameter_reader: ParameterReader,
    ) -> None:
        self.dimension_reader = dimension_reader
        self.parameter_reader = parameter_reader

    def read_dimension(self, definition: DimensionDefinition) -> Dimension:
        """Read a dimension using the dimension reader."""
        return self.dimension_reader.read_dimension(definition)

    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        """Read parameter values using the parameter reader."""
        return self.parameter_reader.read_parameter_values(parameter_name, dims)


def _link_parent(
    dimension: Dimension, definition: DimensionDefinition, parent: Dimension
) -> Dimension:
    """Return a copy of a read dimension, linked to its parent.

    The parent items are converted to the datatype of the parent, if it has one.

    Raises:
        ValueError: If the read dimension has no parent items.
    """
    if dimension.parent_items is None:
        raise ValueError(
            f"Dimension '{definition.name}' has parent '{definition.parent}' in its definition, "
            "but no parent items were read. Give the parent item of each item, e.g. as a second "
            "column in the dimension file."
        )
    parent_items = dimension.parent_items
    if parent.dtype is not None:
        parent_items = [parent.dtype(item) for item in parent_items]
    return Dimension(
        name=dimension.name,
        letter=dimension.letter,
        items=dimension.items,
        dtype=dimension.dtype,
        parent=parent,
        parent_items=parent_items,
    )


def _split_dimension(
    definition: DimensionDefinition, dimensions: dict[str, Dimension]
) -> Dimension:
    """Create a dimension by splitting items of its parent, as given in the definition.

    Args:
        definition: Definition with parent and splitting.
        dimensions: Already created dimensions by letter, including those the definition
            depends on.
    """
    if definition.parent is None or definition.splitting is None:
        raise ValueError(f"Dimension definition '{definition.name}' has no parent or splitting.")
    parent = dimensions[definition.parent]
    splitting = {
        (parent.dtype(item) if parent.dtype is not None else item): dimensions[letter]
        for item, letter in definition.splitting.items()
    }
    split = parent.split(name=definition.name, letter=definition.letter, splitting=splitting)
    return Dimension.model_validate({**dict(split), "dtype": definition.dtype})
