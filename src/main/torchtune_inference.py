# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from load_balancer_client import Client
from omegaconf import DictConfig
from torch import nn
from torchtune import config, generation, training, utils

from src.data.lora_main import get_dataloader
from src.utils.helper import convert_text_to_ymm_list, convert_text_to_ymm_list_v2
from src.utils.submission import (
    create_table,
    db_connect,
    insert_content_in_table,
    table_exists,
)
import itertools
import sys
import time
from typing import Any, Dict, List, Optional, Union

import torch
from omegaconf import DictConfig
from torch import nn

from torchtune import config, generation, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import ChatFormat, InstructTemplate, Message

logger = utils.get_logger("DEBUG")

logger = utils.get_logger("DEBUG")


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = training.get_quantizer_mode(self._quantizer)
        self._db_root = Path(cfg.db_root)
        self._submission_table_name = cfg.submission_table_name
        self.client = Client(server_ip="172.16.3.216", port=8000)

        training.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        self.client.register()
        checkpointer = config.instantiate(cfg.checkpointer)
        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        combined_state_dict = {}
        combined_state_dict.update(ckpt_dict["model"])
        combined_state_dict.update(ckpt_dict["adapter"])

        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=combined_state_dict,  # FIXME: This is a mod
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)
        self._conn = db_connect(self._db_root / "submissions.db")
        if not table_exists(self._conn, self._submission_table_name):
            create_table(self._conn, self._submission_table_name)
        self.dataloader = get_dataloader(cfg.dataset, self.client)
        # Ensure the cache is setup on the right device, with only as many tokens as we need
        if cfg.enable_kv_cache:
            with self._device:
                self._model.setup_caches(
                    batch_size=1,  # TODO: Make this modular
                    dtype=self._dtype,
                    decoder_max_seq_len=16000
                    + cfg.max_new_tokens,  # TODO: Make this modular
                )

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )
        logger.info(f"Model is initialized with precision {self._dtype}.")

        return model

    def convert_prompt_to_tokens(
        self,
        prompt: Union[DictConfig, str],
        chat_format: Optional[ChatFormat],
        instruct_template: Optional[InstructTemplate],
    ) -> List[Message]:
        """
        Either:
        (1) a raw string is passed as the prompt, in which case we call tokenizer.encode directly, or
        (2) a DictConfig is passed as the prompt. In this case there are three possibilities:
            (a) an InstructTemplate is provided. Since instruct templates output a string, we will
                call tokenizer.encode on the output of the instruct template.
            (b) a ChatFormat is provided. Since chat formats output a list of messages, we will
                call tokenizer.tokenize_messages on the output of the chat format.
            (c) neither an InstructTemplate nor a ChatFormat is provided. In this case we will
                convert the DictConfig to a list of messages and call tokenizer.tokenize_messages directly.
        """

        # Should only be chat-style prompt or instruct-style prompt
        if chat_format and instruct_template:
            raise ValueError(
                "Cannot pass both chat format and instruct template for generation"
            )

        # If instruct template is provided, assert that the prompt is a DictConfig
        # and apply it
        if instruct_template:
            if not isinstance(prompt, DictConfig):
                raise ValueError("Cannot apply instruct template to raw string")
            instruct_template = _get_component_from_path(instruct_template)
            prompt = instruct_template.format(prompt)

        # To hit this block, either the raw prompt is a string or an
        # instruct template has been provided to convert it to a string
        if isinstance(prompt, str):
            return self._tokenizer.encode(prompt, add_bos=True, add_eos=False)

        # dict.items() will respect order for Python >= 3.7
        else:
            messages = [Message(role=k, content=v) for k, v in prompt.items()]
            messages += [Message(role="assistant", content="")]
            if chat_format:
                chat_format = _get_component_from_path(chat_format)
                messages = chat_format.format(messages)
            return self._tokenizer.tokenize_messages(messages)[0]

    @torch.inference_mode()
    def generate(self, cfg: DictConfig, tokens, record_ids):
        # if not isinstance(tokens, torch.Tensor):
        #     prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)
        # else:
        #     prompt = tokens.to(self._device)

        custom_generate_next_token = None
        tokens = self.convert_prompt_to_tokens(
            "system\n\nUsing given information about a part, output the cars the it is compatible with: in JSON format [{'FTMNT_YEAR': ..., 'FTMNT_MAKE': ..., 'FTMNT_MODEL': ...}, ...]user\n\nItems Data:Upper Radiator Hose For 1980-1984, 1990-1991 GMC G3500 1981 1982 1983 T676DS<sep>Tags Data:tags=[{'Brand': 'Gates'}, {'CS-SKU': '400:T676DS'}, {'Fitment': 'GMC G3500 Radiator Hose'}, {'Info': 'Flexible Coolant Hose (Standard)'}, {'Manufacturer Warranty': '12 Month Warranty'}, {'Part Number': 'T676DS'}, {'Placement on Vehicle': 'Upper'}, {'Product Name': 'Radiator Hose'}, {'Years': '1980 1981 1982 1983 1984 1990 1991 80 81 82 83 84 90 91'}]<sep>Description Data:b",
            cfg.get("chat_format", None),
            cfg.get("instruct_template", None),
        )
        prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)
        # prompt = self._tokenizer.encode(
        #     dtype=torch.int,
        #     device=self._device,
        # )

        # since quantized model uses torch.compile to get speedup, it needs a warm up / prefill run
        # to
        # get the accurate performance measurement
        if self._quantization_mode is not None:
            logger.info("Starting compilation to improve generation performance ...")
            custom_generate_next_token = torch.compile(
                generation.generate_next_token, mode="max-autotune", fullgraph=True
            )
            t0 = time.perf_counter()
            _ = generation.generate(
                model=self._model,
                prompt=prompt,
                max_generated_tokens=2,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                stop_tokens=self._tokenizer.stop_tokens,
                custom_generate_next_token=custom_generate_next_token,
            )
            t = time.perf_counter() - t0
            logger.info(f"Warmup run for quantized model takes: {t:.02f} sec")
            self._model.reset_caches()

        t0 = time.perf_counter()
        generated_tokens, _ = generation.generate(
            model=self._model,
            prompt=prompt,
            max_generated_tokens=cfg.max_new_tokens,
            pad_id=self._tokenizer.pad_id,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            stop_tokens=self._tokenizer.stop_tokens,
            custom_generate_next_token=custom_generate_next_token,
        )
        generated_tokens = generated_tokens.tolist()
        t = time.perf_counter() - t0

        for i in range(len(generated_tokens)):
            logger.debug("Generated Tokens", json.dumps(generated_tokens[i]))
            logger.debug(
                "Decoded tokens",
                json.dumps(
                    self._tokenizer.decode(generated_tokens[i], truncate_at_eos=False),
                ),
            )

        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(
                    self._model.parameters(), self._model.buffers()
                )
            ]
        )

        tokens_generated = len(generated_tokens[0]) - prompt.size(0)
        tokens_sec = tokens_generated / t
        output_dict_list = []

        for record_id, each_generated_tokens in zip(record_ids, generated_tokens):
            # assistant_index = each_generated_tokens.index(78191)
            # output_tokens = each_generated_tokens[assistant_index + 1 :]
            output_tokens = each_generated_tokens
            output_text = self._tokenizer.decode(
                output_tokens, truncate_at_eos=False, skip_special_tokens=False
            )
            output_dict_list.extend(
                convert_text_to_ymm_list_v2(
                    record_id=record_id, text=output_text.strip()
                )
            )

        for output_dict in output_dict_list:
            insert_content_in_table(
                self._conn, self._submission_table_name, output_dict
            )
        logger.info(f"Output: {output_dict_list}")

        logger.info(
            f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        logger.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    for _, (record_id, tokens) in enumerate(recipe.dataloader):
        recipe.generate(cfg=cfg, tokens=tokens, record_ids=record_id)


if __name__ == "__main__":
    sys.exit(main())
