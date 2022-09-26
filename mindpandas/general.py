"""
This module defines general functions.
"""
import numpy as np
import pandas
from pandas._libs.lib import is_scalar
from pandas.core.dtypes.generic import ABCSeries

from mindpandas.compiler.query_compiler import QueryCompiler as qc
from mindpandas.dataframe import DataFrame
from mindpandas.series import Series


def to_datetime(
        arg,
        errors="raise",
        dayfirst=False,
        yearfirst=False,
        utc=None,
        format=None,
        exact=True,
        unit=None,
        infer_datetime_format=False,
        origin="unix",
        cache=True,
):
    """
    Convert argument to datetime.

    This function converts a scalar, array-like, :class:`Series` or
    :class:`DataFrame`/dict-like to a pandas datetime object.

    Args
        arg : int, float, str, datetime, list, tuple, 1-d array, Series, DataFrame/dict-like
        errors : {'ignore', 'raise', 'coerce'}, default 'raise'
        dayfirst : bool, default False
        yearfirst : bool, default False
        utc : bool, default None
        format_arg : str, default None
        exact : bool, default True
        unit : str, default 'ns'
        infer_datetime_format : bool, default False
        origin : scalar, default 'unix'
        cache : bool, default True

    Returns
        datetime

    Raises
        ParserError
        ValueError
    """

    if not isinstance(arg, (DataFrame, Series)):
        return pandas.to_datetime(
            arg,
            errors=errors,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            utc=utc,
            format=format,
            exact=exact,
            unit=unit,
            infer_datetime_format=infer_datetime_format,
            origin=origin,
            cache=cache,
        )
    output_dataframe = qc.to_datetime(arg,
                                      errors=errors,
                                      dayfirst=dayfirst,
                                      yearfirst=yearfirst,
                                      utc=utc,
                                      format=format,
                                      exact=exact,
                                      unit=unit,
                                      infer_datetime_format=infer_datetime_format,
                                      origin=origin,
                                      cache=cache)
    return output_dataframe


def pivot_table(data,
                values=None,
                index=None,
                columns=None,
                aggfunc='mean',
                fill_value=None,
                margins=False,
                dropna=True,
                margins_name='All',
                observed=False,
                sort=True,
                ):
    """
        Create a spreadsheet-style pivot table as a DataFrame.
        The levels in the pivot table will be stored in MultiIndex objects
        (hierarchical indexes) on the index and columns of the result DataFrame.

        Args
            values : column to aggregate, optional
            index : column, Grouper, array, or list of the previous
            columns : column, Grouper, array, or list of the previous
            aggfunc : function, list of functions, dict, default numpy.mean
            fill_value : scalar, default None
            margins : bool, default False
            dropna : bool, default True
            margins_name : str, default 'All'
            observed : bool, default False
            sort : bool, default True

        Returns
            DataFrame
                An Excel style pivot table.
    """

    def _convert_by(by):
        if by is None:
            by = []
        elif (
                is_scalar(by)
                or isinstance(by, (np.ndarray, pandas.Grouper))
        ):
            by = [by]
        elif (
                isinstance(by, (pandas.Index, ABCSeries))
                or callable(by)
        ):
            raise NotImplementedError("pivot_table only supports by key types that groupby currently supports")
        else:
            by = list(by)
        return by

    def _convert_aggfunc(aggfunc):
        if isinstance(aggfunc, list):
            funcs = []
            for func in aggfunc:
                if isinstance(aggfunc, dict):
                    funcs.append(func)
                else:
                    funcs.append(getattr(func, "__name__", func))
            return funcs
        if isinstance(aggfunc, dict):
            return aggfunc

        return getattr(aggfunc, "__name__", aggfunc)

    index = _convert_by(index)
    columns = _convert_by(columns)

    funcs = _convert_aggfunc(aggfunc)

    if isinstance(aggfunc, list):
        pieces = []
        keys = []
        for func in funcs:
            new_table = qc.pivot_table(
                data,
                values=values,
                index=index,
                columns=columns,
                fill_value=fill_value,
                aggfunc=func,
                margins=margins,
                dropna=dropna,
                margins_name=margins_name,
                observed=observed,
                sort=sort,
            )
            pieces.append(new_table)
            keys.append(getattr(func, "__name__", func))

        table = concat(pieces, keys=keys, axis=1)
        return table

    table = qc.pivot_table(
        data,
        values,
        index,
        columns,
        funcs,
        fill_value,
        margins,
        dropna,
        margins_name,
        observed,
        sort,
    )
    return table


def date_range(start=None,
               end=None,
               periods=None,
               freq=None,
               tz=None,
               normalize: bool = False,
               name=None,
               closed=None,
               **kwargs
               ):
    """
    Return a fixed frequency DatetimeIndex.

    Args
        start : str or datetime-like, optional
        end : str or datetime-like, optional
        periods : int, optional
        freq : str or DateOffset, default 'D'
        tz : str or tzinfo, optional
        normalize : bool, default False
        name : str, default None
        closed : {None, 'left', 'right'}, optional
        inclusive : {"both", "neither", "left", "right"}, default "both"

    Returns
        rng : DatetimeIndex
    """

    output_daterange = qc.default_to_pandas_general(pandas_method="date_range",
                                                    start=start,
                                                    end=end,
                                                    periods=periods,
                                                    freq=freq,
                                                    tz=tz,
                                                    normalize=normalize,
                                                    name=name,
                                                    closed=closed,
                                                    **kwargs)
    return output_daterange


def concat(
        objs,
        axis=0,
        join='outer',
        ignore_index=False,
        verify_integrity=False,
        sort=False,
        keys=None,
        levels=None,
        names=None,
        copy=True,
):
    """
    Concatenate pandas objects along a particular axis with optional set logic
    along the other axes.

    Args
        objs : a sequence or mapping of Series or DataFrame objects
        axis : {0/'index', 1/'columns'}, default 0
        join : {'inner', 'outer'}, default 'outer'
        ignore_index : bool, default False
        keys : sequence, default None
        levels : list of sequences, default None
        names : list, default None
        verify_integrity : bool, default False
        sort : bool, default False
        copy : bool, default True

    Returns
        object, type of objs
    """

    if objs is None:
        raise TypeError("missing input argument")
    if not isinstance(objs, list):
        raise TypeError("input argument is not a list")

    for obj in objs:
        if not isinstance(obj, (DataFrame, Series)):
            msg = (
                f"cannot concatenate object of type '{type(obj)}'; "
                "only Series and DataFrame objs are valid"
            )
            raise TypeError(msg)
    if not isinstance(axis, int):
        raise TypeError(f"axis has to be an integer, got {type(axis)}")
    if axis is None:
        axis = 0

    def _get_axis_number(axis) -> int:
        try:
            return {0: 0, "index": 0, "rows": 0, 1: 1}[axis]
        except KeyError:
            raise ValueError(f"No axis named {axis} for object type {type(obj)}")
    axis = _get_axis_number(axis)

    # When concatenating all Series along the index (axis=0), a Series is returned
    # Otherwise, a DataFrame is returned
    is_series = False
    if all(isinstance(x, Series) for x in objs) and axis == 0:
        is_series = True

    obj_is_series = []
    for obj in objs:
        if isinstance(obj, Series):
            obj_is_series.append(True)
        else:
            obj_is_series.append(False)

    if join is None:
        join = 'outer'

    if not isinstance(ignore_index, bool):
        raise TypeError(f"ignore_index has to be a boolean value, got {type(ignore_index)}")
    if ignore_index is None:
        ignore_index = False

    if verify_integrity is None:
        verify_integrity = False

    if not isinstance(sort, bool):
        raise TypeError(f"sort has to be a boolean value, got {type(sort)}")
    if sort is None:
        sort = False

    if not isinstance(copy, bool):
        raise TypeError(f"copy has to be a boolean value, got {type(copy)}")
    if copy is None:
        copy = True

    output = qc.concat(objs=objs,
                       axis=axis,
                       is_series=is_series,
                       obj_is_series=obj_is_series,
                       join=join,
                       ignore_index=ignore_index,
                       verify_integrity=verify_integrity,
                       sort=sort,
                       keys=keys,
                       levels=levels,
                       names=names,
                       copy=copy,
                       )

    return output
