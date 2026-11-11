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


def test_long_format_further_columns_without_dimension_names():
    data = {"a": LONG["Region"], "b": LONG["Time"], "unit": ["t"] * 4, "value": LONG["value"]}
    with pytest.raises(fd.FlodymDataError) as error:
        from_df(data)
    message = str(error.value)
    found = "Columns found for dimensions: 'a' for 'Region' (by items), 'b' for 'Time' (by items)."
    assert found in message
    assert "Other columns: 'unit', 'value'." in message
    assert "Tip: name the dimension columns" in message


def test_wide_format_ignores_text_columns_right_of_values(caplog):
    wide = {"Region": ["Germany", "France"], 2020: [1.0, 2.0], 2021: [3.0, 4.0]}
    with caplog.at_level(logging.INFO, logger="flodym"):
        result = from_df({**wide, "unit": ["t", "t"], "empty": [None, None]})
    assert np.array_equal(result.values, EXPECTED)
    assert "Ignoring columns right of the value columns of 'p': 'unit', 'empty'" in caplog.text


def test_wide_format_numeric_columns_right_of_values():
    wide = {"Region": ["Germany", "France"], 2020: [1.0, 2.0], 2021: [3.0, 4.0], 2022: [5, 6]}
    with pytest.raises(fd.FlodymDataError, match="right of them contain numbers: 2022"):
        from_df(wide)
    assert np.array_equal(from_df(wide, allow_extra_values=True).values, EXPECTED)


def test_wide_format_hints():
    with pytest.raises(fd.FlodymDataError, match="Items without column: 2021"):
        from_df({"Region": ["Germany", "France"], 2020: [1.0, 2.0], 2022: [3.0, 4.0]})
    with pytest.raises(fd.FlodymDataError, match="Further columns left of them: 'unit'"):
        from_df({"Region": ["Germany", "France"], "unit": ["t", "t"], 2020: [1, 2], 2021: [3, 4]})


def test_column_hints():
    data = {"unit": ["t"] * 4, "Regoin": LONG["Region"], "Time": LONG["Time"], "value": [1] * 4}
    with pytest.raises(fd.FlodymDataError, match="'Regoin' is not named .* Did you mean 'Region'"):
        from_df(data)
    data = {"a": ["Germany", "Frnace"] * 2, "b": LONG["Time"], "value": LONG["value"]}
    with pytest.raises(fd.FlodymDataError) as error:
        from_df(data)
    message = str(error.value)
    assert "Column 'a' could be dimension 'Region'" in message
    assert "'Frnace' (index 1, 3; did you mean 'France'?)" in message
    assert "Column 'b' has the items of dimension 'Time', but is right of a column" in message
