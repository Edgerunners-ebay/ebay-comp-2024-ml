import json
import time
from typing import Any

import numpy as np
import pandas as pd
from litellm.exceptions import RateLimitError, ServiceUnavailableError
from torch.utils.data import Dataset
from torchtune.data import (
    CROSS_ENTROPY_IGNORE_IDX,
    JSONToMessages,
)
from torchtune.models.llama3 import Llama3Tokenizer
from torchtune.utils import logging

from src.data.llm_preprocess import (
    preprocess_desc_data,
    preprocess_desc_data_groq_w_name,
)

from .main import Data

logger = logging.get_logger("DEBUG")


class EbayDataset(Dataset):
    def __init__(self, cfg_dataset, return_record_id=False, is_inference=False):
        self.data = Data(cfg_dataset)
        self.tokenizer_path = cfg_dataset.tokenizer_path
        self.max_seq_len = cfg_dataset.max_seq_len_custom
        self._return_record_id = return_record_id
        self._is_inference = cfg_dataset.get("is_inference", False)
        logger.debug(f"Dataset in inference mode: {self._is_inference}")
        self._system_prompt = """
        Using the given information about a part, predict all the cars that this part is compatible with, regardless of where this information appears in the data provided.
        Instructions:
        - Search through Items Data, Tags Data, and Description Data to find any mentions of compatible vehicles.
        - Extract the fitment `year`, `make`, `model` for each compatible vehicle.
        - Ignore irrelevant information not related to vehicle compatibility.
        - Use what could be the official names for makes and models
        - Use model name as reference and predict make if make is not available
        - Give years in the format of YYYY
        - Include only if absolutely sure about the compatibility
        - Output the results in a structured JSON way
        - {"make": {"model": [year, year, year]}}
        Eg: {"toyota": {"corolla": [2010, 2011, 2012], "camry": [2010, 2011, 2012]}}
        """
        self.is_inference = is_inference

    def __len__(self):
        return 5000000

    def __getitem__(self, index) -> Any:
        self.data.get(index)
        if self._return_record_id:
            return index, self._prepare_sample(self.data)
        return self._prepare_sample(self.data)

    def _prepare_sample(self, sample):
        if self._is_inference:
            return self._simple_prompt(sample)

        transformed_sample = self._convert_data_to_messages(sample)

        tokenized_dict = self._convert_messages_to_tokens(transformed_sample)

        if not ("tokens" in tokenized_dict and "mask" in tokenized_dict):
            keys_str = ", ".join(tokenized_dict.keys())
            error_message = (
                "model_transform returned the following keys: "
                f"{keys_str}. Must return 'tokens' and 'mask' as keys."
            )
            raise ValueError(error_message)

        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        tokenized_dict["labels"] = list(
            np.where(
                tokenized_dict["mask"],
                CROSS_ENTROPY_IGNORE_IDX,
                tokenized_dict["tokens"],
            )
        )
        print("length of tokens", len(tokenized_dict["tokens"]))
        assert len(tokenized_dict["tokens"]) == len(tokenized_dict["labels"])
        return tokenized_dict

    def _simple_prompt(self, sample):
        return {
            "system": self._system_prompt,
            "user": self._convert_sample_to_input_old(sample),
        }

    def _convert_data_to_messages(self, sample):
        input_text = self._convert_sample_to_input_old(sample)
        output_text = ""
        if not self._is_inference:
            output_text = self._convert_df_to_json_efficient(sample.ftmnt_data)
        message_transform = JSONToMessages(
            train_on_input=True,
            column_map=None,
            new_system_prompt=None,
        )
        sample = {
            "messages": [
                {
                    "role": "system",
                    "content": f"{self._system_prompt}",
                },
                {"role": "user", "content": input_text},
            ]
        }
        if not self._is_inference:
            sample["messages"].append(
                {
                    "role": "assistant",
                    "content": output_text if not self._is_inference else "",
                },
            )
        updated_messages = message_transform(sample)
        messages_list = [x.content for x in updated_messages["messages"]]
        logger.debug(f"Updated messages: {messages_list}")
        return updated_messages

    def _convert_messages_to_tokens(self, messages):
        _tokenizer = Llama3Tokenizer(
            path=self.tokenizer_path, max_seq_len=self.max_seq_len
        )
        return _tokenizer(messages, inference=self._is_inference)

    def _convert_sample_to_input_old(self, data):
        return f"Items Data:{data.items_data} \n Tags Data:{data.tags_data} \n Description Data:{data.desc_data}"

    def _convert_sample_to_input(self, data):
        while True:
            try:
                desc_data_preprocessed = preprocess_desc_data_groq_w_name(
                    text=data.desc_data[:30000], _model="groq/gemma2-9b-it"
                )
                break
            except ServiceUnavailableError:
                logger.debug("Something messed up on Groq side, waiting for a minute")
                time.sleep(60)
                continue

            except RateLimitError:
                logger.debug("Hit the rate limit, waiting out a minute")
                time.sleep(60)
                continue
            except Exception as e:
                logger.debug("some error", e)
                desc_data_preprocessed = data.desc_data
                break

        return f"Items Data:{data.items_data} \n Tags Data:{data.tags_data} \n Description Data:{desc_data_preprocessed}"

    def _convert_df_to_output(self, df: pd.DataFrame):
        output = ""
        length = len(df.values)
        for idx, val in enumerate(df.values):
            output += "<sep>".join([str(x) for x in val])
            if idx < length - 1:
                output += "<next>"
        return output

    def _convert_df_to_json(self, df):
        output = []
        for _, val in enumerate(df.values):
            output.append({str(x): str(y) for x, y in zip(df.columns, val)})
        return json.dumps(output)

    def _convert_df_to_json_efficient(self, df):
        output = {}
        for _, val in enumerate(df.values):
            if val[1] not in output:
                output[val[1]] = {}
            if val[2] not in output[val[1]]:
                output[val[1]][val[2]] = []
            output[val[1]][val[2]].append(val[0])
        return json.dumps(output)
