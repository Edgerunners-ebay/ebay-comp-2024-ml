import base64
import json
import re
from pathlib import Path
from typing import Dict, List

import pyarrow.dataset as ds
import pydantic
from bs4 import BeautifulSoup
from pydantic import BaseModel

from .html_preprocess import html_reducer


class Info(BaseModel, arbitrary_types_allowed=True):
    desc: ds.Dataset
    fitment: ds.Dataset
    tags: ds.Dataset
    items: ds.Dataset


class Tags(pydantic.BaseModel):
    tags: List[Dict]


class Data:
    def __init__(self, cfg):
        desc_ds = ds.dataset(
            [
                Path(cfg["EBAY_ROOT_HPC"]) / "optimised_files" / "desc_0.parquet",
                Path(cfg["EBAY_ROOT_HPC"]) / "optimised_files" / "desc_1.parquet",
                Path(cfg["EBAY_ROOT_HPC"]) / "optimised_files" / "desc_2.parquet",
                Path(cfg["EBAY_ROOT_HPC"]) / "optimised_files" / "desc_3.parquet",
            ],
            format="parquet",
        )

        fitment_ds = ds.dataset(
            Path(cfg["EBAY_ROOT_HPC"]) / "optimised_files" / "ftmnt_train.parquet",
        )

        tags_ds = ds.dataset(
            Path(cfg["EBAY_ROOT_HPC"]) / "optimised_files" / "tags.parquet",
        )

        items_ds = ds.dataset(
            Path(cfg["EBAY_ROOT_HPC"]) / "optimised_files" / "items.parquet",
        )

        self.info = Info(desc=desc_ds, fitment=fitment_ds, tags=tags_ds, items=items_ds)
        assert isinstance(self.info, Info), "info should be of type Info"
        self.ftmnt_data = None
        self.desc_data = None
        self.tags_data = None
        self.items_data = None

    def get(self, record_id: int):
        """
        Get data from an individual record_id
        """
        record_filtering = ds.field("RECORD_ID") == record_id

        fitment_columns_to_read = ["FTMNT_YEAR", "FTMNT_MAKE", "FTMNT_MODEL"]
        self.ftmnt_data = self.info.fitment.to_table(
            filter=record_filtering, columns=fitment_columns_to_read
        ).to_pandas()

        desc_columns_to_read = ["ITEM_DESC"]
        desc_data = self.info.desc.to_table(
            filter=record_filtering, columns=desc_columns_to_read
        ).to_pandas()["ITEM_DESC"]

        if desc_data is None:
            desc_data = [""]

        self.desc_data = base64.b64decode(s=" ".join(desc_data))

        # Manual Preprocess desc data
        try:
            self.desc_data = html_reducer(self.desc_data)
        except Exception as e:
            print(f"Error in html_reducer: {e}")
            self.desc_data = (
                BeautifulSoup(self.desc_data, "html.parser")
                .get_text()
                .replace("\n", "")
                .replace("\r", "")
                .replace("\t", "")
            )
            self.desc_data = re.sub(r"\s+", " ", self.desc_data)

        tags_columns_to_read = ["ITEM_TAGS"]
        self.tags_data = Tags(
            tags=json.loads(
                " ".join(
                    self.info.tags.to_table(
                        filter=record_filtering, columns=tags_columns_to_read
                    ).to_pandas()["ITEM_TAGS"]
                )
            )
        )

        items_columns_to_read = ["CATEGORY", "ITEM_TITLE"]
        self.items_data = " ".join(
            self.info.items.to_table(
                filter=record_filtering, columns=items_columns_to_read
            ).to_pandas()["ITEM_TITLE"]
        )
