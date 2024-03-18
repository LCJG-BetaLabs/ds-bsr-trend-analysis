import os
import pandas as pd
from typing import List, Optional, Union
from databricks.sdk.runtime import spark

IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ
ENVIRONMENT = os.environ.get("ENVIRONMENT", "prd")

TREND_CLASSIFICATION_RESULT = f"lc_{ENVIRONMENT}.ml_trend_analysis_silver.trend_classification"
SALES = f"lc_{ENVIRONMENT}.ml_trend_analysis_silver.sales"
WEEK_COVERAGE = f"lc_{ENVIRONMENT}.ml_trend_analysis_silver.week_coverage"
VPN_STYLE_MAP = f"lc_{ENVIRONMENT}.ml_trend_analysis_silver.vpn_style_map"
VPN_INFO = f"lc_{ENVIRONMENT}.ml_trend_analysis_silver.vpn_info"


def uc_table_exists(full_table_name) -> bool:
    return spark.catalog.tableExists(full_table_name)


def read_uc_table(
    full_table_name: str,
    columns: Optional[List[str]] = None,
) -> "pyspark.sql.dataframe.DataFrame":
    """
    Attempts to read table from UC.
    """
    df = spark.table(full_table_name)
    if columns:
        df = df.select(*columns)
    return df


def write_uc_table(
    full_table_name: str,
    df: Union[pd.DataFrame, "pyspark.sql.dataframe.DataFrame"],
    mode: str = "merge",
    primary_keys: Optional[Union[str, List[str]]] = None,
    **kwargs,
):
    if mode == "merge":
        print(
            """'merge' not support in `pyspark.sql.DataFrameWriter.saveAsTable`, using 'append' instead"""
        )
        mode = "append"
    _write_uc_table(full_table_name, df, mode, primary_keys, **kwargs)


def _write_uc_table(
    full_table_name: str,
    df: Union[pd.DataFrame, "pyspark.sql.dataframe.DataFrame"],
    mode: str = "append",
    primary_keys: Optional[Union[str, List[str]]] = None,
    comment: str = None,
    **kwargs,
):
    if not IS_DATABRICKS:
        raise ValueError("Cannot write to UC when not running on Databricks")

    if isinstance(df, pd.DataFrame):
        df = spark.createDataFrame(df)

    if uc_table_exists(full_table_name):
        print(f"Found table {full_table_name}, inserting to table.")
        df.write.format("delta").saveAsTable(f"{full_table_name}", mode=mode, **kwargs)
    else:
        print(f"Table {full_table_name} is not found, Creating table.")
        owner = "betalabsds"
        df.write.format("delta").mode("overwrite").saveAsTable(
            f"{full_table_name}", **kwargs
        )
        spark.sql(f"ALTER TABLE {full_table_name} OWNER TO {owner}")
        print(f"Created table {full_table_name}")
        if comment:
            spark.sql(f"COMMENT ON TABLE {full_table_name} IS '{comment}'")
        if primary_keys:
            if isinstance(primary_keys, str):
                primary_keys = list(primary_keys)
            # refer to https://docs.databricks.com/en/machine-learning/feature-store/uc/feature-tables-uc.html#use-an-existing-delta-table-in-unity-catalog-as-a-feature-table
            for col in primary_keys:
                spark.sql(
                    f"""ALTER TABLE {full_table_name} ALTER COLUMN {col} SET NOT NULL"""
                )
            spark.sql(
                f"""ALTER TABLE {full_table_name} ADD CONSTRAINT {full_table_name.split(".")[-1]}_pk PRIMARY KEY ({",".join(primary_keys)})"""
            )
