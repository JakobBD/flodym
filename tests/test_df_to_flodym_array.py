"""Tests of the error messages and of the handling of further columns when reading DataFrames."""

import logging

import numpy as np
import pandas as pd
import pytest

import flodym as fd

REGION = fd.Dimension(name="Region", letter="r", items=["Germany", "France"], dtype=str)
TIME = fd.Dimension(name="Time", letter="t", items=[2020, 2021], dtype=int)
DIMS = fd.DimensionSet(dim_list=[REGION, TIME])
LONG = {
    "Region": ["Germany", "France", "Germany", "France"],
    "Time": [2020, 2020, 2021, 2021],
    "value": [1.0, 2.0, 3.0, 4.0],
}
EXPECTED = np.array([[1.0, 3.0], [2.0, 4.0]])


def from_df(data: dict, **kwargs) -> fd.Parameter:
    return fd.Parameter.from_df(dims=DIMS, df=pd.DataFrame(data), name="p", **kwargs)


def test_error_names_array_and_dimensions():
    with pytest.raises(fd.FlodymDataError) as error:
        from_df({**LONG, "Region": ["Germany", "Frnace", "Germany", "Frnace"]})
    message = str(error.value)
    assert message.startswith(
        "While setting values of Parameter 'p' with dimensions Region (r), Time (t) from a "
        "DataFrame:\n  Column 'Region' has items that are not in the dimension: 'Frnace' "
        "(index 1, 3; did you mean 'France'?)."
    )
    assert "allow_extra_values=True" in message
    assert isinstance(error.value, ValueError)


def test_error_for_unnamed_array_recommends_name():
    with pytest.raises(fd.FlodymDataError, match=r"unnamed Parameter .* \(give name=\.\.\."):
        fd.Parameter.from_df(dims=DIMS, df=pd.DataFrame({**LONG, "value": [1, 2, "x", 4]}))


def test_non_numeric_value_names_entry_and_row():
    with pytest.raises(fd.FlodymDataError, match=r"not numbers: 'x' \(index 2\)"):
        from_df({**LONG, "value": [1, 2, "x", 4]})


def test_item_that_cannot_be_converted_to_dtype():
    with pytest.raises(fd.FlodymDataError, match=r"type int .*'y2021' \(index 2\)"):
        from_df({**LONG, "Time": ["2020", "2020", "y2021", "2021"]})


def test_items_of_other_type_get_dtype_hint():
    time_without_dtype = fd.Dimension(name="Time", letter="t", items=[2020, 2021])
    dims = fd.DimensionSet(dim_list=[REGION, time_without_dtype])
    df = pd.DataFrame({**LONG, "Time": ["2020", "2020", "2021", "2021"]})
    with pytest.raises(fd.FlodymDataError, match="have type str, but .* have type int"):
        fd.Parameter.from_df(dims=dims, df=df)


def test_empty_cells():
    with pytest.raises(fd.FlodymDataError, match=r"Column 'Region' has empty cells \(index 1\)"):
        from_df({**LONG, "Region": ["Germany", None, "Germany", "France"]})
    with pytest.raises(fd.FlodymDataError, match=r"\('France', 2020\) \(index 1\)"):
        from_df({**LONG, "value": [1.0, None, 3.0, 4.0]})


def test_duplicates_and_missing_values():
    duplicates = r"(?s)more than once.*\('Germany', 2020\) \(index 0, 1\)"
    with pytest.raises(fd.FlodymDataError, match=duplicates):
        from_df({**LONG, "Region": ["Germany", "Germany", "Germany", "France"]})
    with pytest.raises(fd.FlodymDataError, match=r"missing for 1 of 4 .*\('France', 2021\)"):
        from_df({key: values[:3] for key, values in LONG.items()})


def test_empty_dataframe():
    with pytest.raises(fd.FlodymDataError, match="no columns"):
        fd.Parameter.from_df(dims=DIMS, df=pd.DataFrame())


def test_empty_rows_and_unnamed_empty_columns_are_dropped():
    data = {key: [*values, None] for key, values in LONG.items()}
    data["Unnamed: 3"] = [None] * 5
    assert np.array_equal(from_df(data).values, EXPECTED)


def test_integer_and_mixed_column_names():
    # integer name of the value column
    data = {"Region": LONG["Region"], "Time": LONG["Time"], 0: LONG["value"]}
    assert np.array_equal(from_df(data).values, EXPECTED)
    # wide format with integer and string year columns
    wide = {"Region": ["Germany", "France"], 2020: [1.0, 2.0], "2021": [3.0, 4.0]}
    assert np.array_equal(from_df(wide).values, EXPECTED)


def test_long_format_ignores_further_columns_if_unambiguous(caplog):
    data = {"comment": ["a"] * 4, **LONG, "unit": ["t"] * 4, "age": [2010] * 4}
    with caplog.at_level(logging.INFO, logger="flodym"):
        assert np.array_equal(from_df(data).values, EXPECTED)
    assert "Ignoring columns of 'p'" in caplog.text
    assert "'comment', 'unit', 'age'" in caplog.text


def test_long_format_value_column():
    # the only further column is the value column, whatever its name
    data = {"Region": LONG["Region"], "Time": LONG["Time"], "amount": LONG["value"]}
    assert np.array_equal(from_df(data).values, EXPECTED)
    # a column without header (e.g. the index written by to_csv) is not taken as value column
    data["Unnamed: 0"] = [0, 1, 2, 3]
    assert np.array_equal(from_df(data).values, EXPECTED)
    with pytest.raises(fd.FlodymDataError, match="several further columns.*Name the value column"):
        from_df({**data, "unit": ["t"] * 4})


def test_dimension_columns_must_be_named():
    data = {"a": LONG["Region"], "b": LONG["Time"], "unit": ["t"] * 4, "value": LONG["value"]}
    with pytest.raises(fd.FlodymDataError) as error:
        from_df(data)
    message = str(error.value)
    assert "Long format assumed, as there is a column 'value'." in message
    assert "No column for dimension(s) 'Region' (r), 'Time' (t)." in message
    assert "- Column 'a' holds items of 'Region'. Name it 'Region' or 'r'." in message
    assert "- Column 'b' holds items of 'Time'. Name it 'Time' or 't'." in message


def test_wide_format_ignores_columns_without_numbers(caplog):
    wide = {"Region": ["Germany", "France"], "unit": ["t", "t"], 2020: [1.0, 2.0], 2021: [3, 4]}
    with caplog.at_level(logging.INFO, logger="flodym"):
        result = from_df({**wide, "comment": ["x", None], "Unnamed: 5": [0, 1]})
    assert np.array_equal(result.values, EXPECTED)
    assert "Ignoring columns of 'p' which are neither dimension nor value columns: 'unit', " in (
        caplog.text
    )


def test_wide_format_numeric_columns_that_are_not_items():
    wide = {"Region": ["Germany", "France"], 2020: [1.0, 2.0], 2021: [3.0, 4.0], 2022: [5, 6]}
    with pytest.raises(fd.FlodymDataError) as error:
        from_df(wide)
    message = str(error.value)
    assert "Wide format over 'Time' assumed, as there is no column 'value'" in message
    assert "but their names are not items of 'Time': 2022." in message
    assert np.array_equal(from_df(wide, allow_extra_values=True).values, EXPECTED)
    with pytest.raises(fd.FlodymDataError, match="header '2020' seems to occur twice"):
        from_df({"Region": ["Germany", "France"], "2020": [1, 2], "2021": [3, 4], "2020.1": [5, 6]})


def test_wide_format_items_without_column():
    wide = {"Region": ["Germany", "France"], 2020: [1.0, 2.0]}
    with pytest.raises(fd.FlodymDataError, match="Items of 'Time' without column: 2021"):
        from_df(wide)
    result = from_df(wide, allow_missing_values=True)
    assert np.array_equal(result.values, [[1.0, 0.0], [2.0, 0.0]])


def test_misspelled_dimension_column_is_not_read_as_wide_format():
    data = {"Regoin": LONG["Region"], "Time": LONG["Time"], "amount": LONG["value"]}
    with pytest.raises(fd.FlodymDataError) as error:
        from_df({**data, "Regoin": ["Germany", "Frnace"] * 2})
    message = str(error.value)
    assert "No column for dimension(s) 'Region' (r)." in message
    assert "Column 'Regoin' is not named like a dimension. Did you mean 'Region'?" in message


def test_header_hints():
    with pytest.raises(fd.FlodymDataError, match="the data have no header row"):
        from_df({0: LONG["Region"], 1: LONG["Time"], 2: LONG["value"]})
    # the first data row was read as header
    data = {
        "Germany": ["France", "Germany", "France"],
        "2020": [2020, 2021, 2021],
        "1.0": [2, 3, 4],
    }
    with pytest.raises(fd.FlodymDataError, match="The first row seems to be data, not a header"):
        from_df(data)
    data = {"Unnamed: 0": ["Germany", "France"], "Unnamed: 1": [1, 2], "Title": [None, None]}
    with pytest.raises(fd.FlodymDataError, match="Most columns have no header"):
        from_df(data)
    indexed = pd.DataFrame({"Time": LONG["Time"], "value": LONG["value"]}, index=LONG["Region"])
    with pytest.raises(fd.FlodymDataError, match="The index holds items of 'Region'"):
        fd.Parameter.from_df(dims=DIMS, df=indexed)
    indexed = pd.DataFrame(LONG).set_index(["Region", "Time"])
    assert np.array_equal(fd.Parameter.from_df(dims=DIMS, df=indexed).values, EXPECTED)


@pytest.mark.parametrize("field", ["2.5", 2.5])
def test_single_field_is_value_of_array_with_single_value(field):
    single = fd.Dimension(name="Scenario", letter="s", items=["base"])
    for dims in [fd.DimensionSet.empty(), fd.DimensionSet(dim_list=[single])]:
        df = pd.DataFrame(columns=[field])
        assert fd.Parameter.from_df(dims=dims, df=df).values.flatten().tolist() == [2.5]
    with pytest.raises(fd.FlodymDataError, match="no rows"):
        fd.Parameter.from_df(dims=fd.DimensionSet.empty(), df=pd.DataFrame(columns=["value"]))
