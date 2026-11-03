import polars as pl

FEATURES = ["item", "brand", "user_id", "region", "socdem_cluster"]


def normalize_events(
    frame: pl.DataFrame, domain: str, items: pl.DataFrame | None
) -> pl.DataFrame:
    frame = frame.with_columns(
        pl.col("user_id").cast(pl.String).alias("client_id"),
        pl.col("timestamp")
        .dt.total_microseconds()
        .cast(pl.Datetime("us"))
        .cast(pl.Datetime("ns")),
    )
    if items is not None:
        frame = (
            frame.with_columns(pl.col("item_id").cast(pl.String))
            .join(
                items,
                on="item_id",
                how="left",
                validate="m:1",
                maintain_order="left",
            )
            .with_columns(
                pl.concat_str(pl.lit(f"{domain}:"), pl.col("item_id")).alias("item")
            )
        )
    else:
        frame = frame.with_columns(
            pl.lit(None, pl.String).alias("item"),
            pl.lit("transaction").alias("action_type"),
        )
    return frame.select(
        "client_id",
        "timestamp",
        "item",
        pl.col("brand_id").cast(pl.String).alias("brand"),
        pl.concat_str(pl.lit(f"{domain}-"), pl.col("action_type")).alias("event"),
    )
