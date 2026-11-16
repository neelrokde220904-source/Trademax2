"""Safe LangGraph checkpoint serialization for market-data DataFrames."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

_DATAFRAME_MARKER = "__india_trader_dataframe_v2__"


class TradingCheckpointSerializer(JsonPlusSerializer):
    """Serialize supported DataFrames losslessly without pickle."""

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return super().dumps_typed(self._encode(obj))

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return self._decode(super().loads_typed(data))

    @classmethod
    def _encode_series(cls, series: pd.Series) -> dict[str, Any]:
        dtype = series.dtype
        if isinstance(dtype, pd.DatetimeTZDtype):
            return {
                "kind": "datetime_tz", "unit": dtype.unit, "timezone": str(dtype.tz),
                "values": series.array.asi8.tobytes(),
            }
        if not isinstance(dtype, np.dtype):
            raise TypeError(f"Unsupported checkpoint column dtype: {dtype}")
        if dtype.kind in "biufmM":
            return {"kind": "numpy", "dtype": dtype.str, "values": series.to_numpy().tobytes()}
        if dtype.kind == "O":
            values = series.tolist()
            if not all(item is None or type(item) in (str, bool, int, float) for item in values):
                raise TypeError("Unsupported object value in checkpoint DataFrame")
            return {"kind": "object", "values": values}
        raise TypeError(f"Unsupported checkpoint column dtype: {dtype}")

    @classmethod
    def _decode_series(cls, data: dict[str, Any]) -> pd.Series:
        kind = data["kind"]
        if kind == "object":
            return pd.Series(data["values"], dtype=object)
        if kind == "datetime_tz":
            values = np.frombuffer(data["values"], dtype=np.int64)
            index = pd.to_datetime(values, unit=data["unit"], utc=True)
            return pd.Series(index.tz_convert(data["timezone"]).as_unit(data["unit"]))
        if kind == "numpy":
            dtype = np.dtype(data["dtype"])
            if dtype.kind not in "biufmM":
                raise ValueError("Invalid checkpoint array dtype")
            return pd.Series(np.frombuffer(data["values"], dtype=dtype).copy())
        raise ValueError("Invalid checkpoint series kind")

    @classmethod
    def _encode(cls, value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            if not value.columns.is_unique or not all(isinstance(c, str) for c in value.columns):
                raise TypeError("Checkpoint DataFrame columns must be unique strings")
            if value.columns.name is not None and not isinstance(value.columns.name, str):
                raise TypeError("Unsupported checkpoint column index name")
            if value.index.name is not None and not isinstance(value.index.name, str):
                raise TypeError("Unsupported checkpoint index name")
            if isinstance(value.index, pd.MultiIndex):
                raise TypeError("MultiIndex checkpoints are not supported")
            if isinstance(value.index, pd.RangeIndex):
                index = {"kind": "range", "start": value.index.start,
                         "stop": value.index.stop, "step": value.index.step}
            else:
                index = {"kind": "values", "data": cls._encode_series(value.index.to_series(index=None))}
            index["name"] = value.index.name
            return {_DATAFRAME_MARKER: {
                "columns": list(value.columns), "columns_name": value.columns.name,
                "index": index,
                "data": [cls._encode_series(value.iloc[:, i]) for i in range(len(value.columns))],
            }}
        if isinstance(value, dict):
            if _DATAFRAME_MARKER in value:
                raise ValueError("Reserved checkpoint marker appears in graph state")
            return {key: cls._encode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._encode(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._encode(item) for item in value)
        return value

    @classmethod
    def _decode(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if _DATAFRAME_MARKER in value:
                if len(value) != 1 or not isinstance(value[_DATAFRAME_MARKER], dict):
                    raise ValueError("Invalid checkpoint DataFrame")
                data = value[_DATAFRAME_MARKER]
                index_data = data["index"]
                if index_data["kind"] == "range":
                    index = pd.RangeIndex(index_data["start"], index_data["stop"],
                                          index_data["step"], name=index_data["name"])
                elif index_data["kind"] == "values":
                    index = pd.Index(cls._decode_series(index_data["data"]).array,
                                     name=index_data["name"])
                else:
                    raise ValueError("Invalid checkpoint index kind")
                if len(data["columns"]) != len(data["data"]):
                    raise ValueError("Invalid checkpoint column count")
                series = [cls._decode_series(column) for column in data["data"]]
                if any(len(column) != len(index) for column in series):
                    raise ValueError("Invalid checkpoint row count")
                frame = pd.concat(series, axis=1) if series else pd.DataFrame(index=range(len(index)))
                frame.columns = pd.Index(data["columns"], name=data["columns_name"])
                frame.index = index
                return frame
            return {key: cls._decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._decode(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._decode(item) for item in value)
        return value
