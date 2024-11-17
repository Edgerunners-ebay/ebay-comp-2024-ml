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
from typing import Any, Dict, List

import torch
from load_balancer_client import Client
from omegaconf import DictConfig, OmegaConf
from torchtune import config, training, utils
from torchtune.data import Message, load_image, padded_collate_tiled_images_and_mask
from torchtune.generation import sample
from torchtune.modules.transforms import Transform
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX

from src.data.lora_main import get_dataloader
from src.utils.helper import calculate_varentropy_logsoftmax
from src.utils.submission import (
    create_table,
    create_table_w_entropy_varentropy,
    db_connect,
    insert_content_in_table,
    insert_content_in_table_w_entropy_varentropy,
    table_exists,
)


class SingleTurnYAMLToMessages(Transform):
    """
    Converts a single turn conversation in YAML format to a list of messages.

    Expects the YAML to look like:
        system: You are a helpful AI assistant.
        user: What is the capital of France?

    or if it includes an image:
        system: You are a helpful AI assistant.
        user:
            image: url or path_to_image
            text: Describe the image in detail.
    """

    def __call__(self, prompt: Dict[str, Any]) -> List[Message]:
        messages = []

        # Iterate through roles and add content
        for role, content in prompt.items():
            if isinstance(content, str):
                new_content = [{"type": "text", "content": content}]
            else:
                assert (
                    "image" in content.keys()
                ), "Multiple entries per role expect an image key"
                image_loc = content["image"]
                image = load_image(image_loc)
                new_content = [
                    {"type": "image", "content": image},
                    {"type": "text", "content": content["text"]},
                ]
            messages.append(Message(role=role, content=new_content))

        # Finally, add an empty assistant message to kick-start generation
        messages.append(Message(role="assistant", content=""))
        return messages


from torch.nn.utils.rnn import pad_sequence


def pad_to_max_seq_len(batch, padding_idx=0, max_seq_len=16000):
    input_ids = pad_sequence(
        [torch.tensor(x["tokens"]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )

    input_ids_seq_len = input_ids.shape[-1]
    input_ids = torch.F.pad(
        input_ids,
        (0, max_seq_len - input_ids_seq_len),
        value=padding_idx,
    )
    return {"tokens": input_ids.long()}


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.
    This works for text-only generation and image-text generation.

    This *does not* currently support the following features:
        - torch.compile
        - quantization through torchao
        - multi-GPU generation
        - batch generation
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._logger = utils.get_logger(cfg.log_level)
        training.set_seed(seed=cfg.seed)
        self._setup_submission(cfg)
        self._setup_dataloader(cfg)

    def _setup_dataloader(self, cfg):
        self.dataloader = get_dataloader(
            cfg.dataset, self.client, batch_size=1, is_inference=True
        )

    def _setup_submission(self, cfg):
        self._db_root = Path(cfg.db_root)
        self.client = Client(server_ip="172.16.3.216", port=8000)
        self._conn = db_connect(self._db_root / "submissions.db")
        self._submission_table_name = cfg.submission_table_name
        if not table_exists(self._conn, self._submission_table_name):
            create_table_w_entropy_varentropy(self._conn, self._submission_table_name)

    def _submit_to_db(self, record_id, decoded_content):
        try:
            json_content = json.loads(decoded_content)
        except Exception as e:
            self._logger.error(
                f"Error in converting to json: {e}, maybe the max tokens is too low?"
            )
            try:
                modified_decoded_content = decoded_content.split("}")[:-1] + ["]"]
                modified_decoded_content = "}".join(modified_decoded_content)
                json_content = json.loads(modified_decoded_content)
            except Exception as e:
                self._logger.error(
                    f"Split method failed: {e}, something messed up, check manually, {modified_decoded_content}, trying to split once more"
                )
                try:
                    modified_decoded_content = decoded_content.split("}")[:-2] + ["]"]
                    modified_decoded_content = "}".join(modified_decoded_content)
                    json_content = json.loads(modified_decoded_content)
                except Exception as e:
                    self._logger.error(
                        f"Split method failed: {e}, something messed up, check manually, {modified_decoded_content}, can't fix this anymore"
                    )
                    return
        for _, content in enumerate(json_content):
            temp_dict = {}
            temp_dict["RECORD_ID"] = (
                record_id[0] if isinstance(record_id, list) else record_id
            )
            temp_dict.update(content)
            insert_content_in_table_w_entropy_varentropy(
                self._conn, self._submission_table_name, temp_dict
            )

    def setup(self, cfg: DictConfig) -> None:
        """Setup the model and transforms."""
        # Load checkpointer and state_dict
        _checkpointer = config.instantiate(cfg.checkpointer)
        _ckpt_dict = _checkpointer.load_checkpoint()

        # Instantiate model
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg.model)

        combined_state_dict = {}
        combined_state_dict.update(_ckpt_dict["model"])
        combined_state_dict.update(_ckpt_dict["adapter"])

        model.load_state_dict(combined_state_dict)
        self.model = model
        self._logger.info(f"Model was initialized with precision {self._dtype}.")

        # Instantiate transforms
        self.model_transform = config.instantiate(cfg.tokenizer)
        self.to_messages = SingleTurnYAMLToMessages()

    def log_metrics(self, total_time: int, tokens_per_second: float) -> None:
        """Logs the following metrics: total time for inference, tokens/sec,
        bandwidth achieved, and max memory allocated.

        Feel free to modify this function to log additional metrics.
        """
        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(self.model.parameters(), self.model.buffers())
            ]
        )
        self._logger.info(
            f"Time for inference: {total_time:.02f} sec total, {tokens_per_second:.02f} tokens/sec"
        )
        self._logger.info(
            f"Bandwidth achieved: {model_size * tokens_per_second / 1e9:.02f} GB/s"
        )
        self._logger.info(
            f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB"
        )

    @torch.inference_mode()
    def generate(
        self, cfg: DictConfig, prompt: Dict[str, Any], record_id: List[int]
    ) -> None:
        """The main entry point for generating tokens from a prompt."""
        # 1. Convert input to messages
        # messages = self.to_messages(OmegaConf.to_container(prompt))
        messages = self.to_messages(prompt[0])
        is_multimodal_input = any([m.contains_media for m in messages])

        # 2. Apply model transform
        model_inputs = self.model_transform({"messages": messages}, inference=True)
        seq_len = 16000
        total_response_length = seq_len + cfg.max_new_tokens

        # 3. Setup KV cache
        with self._device:
            self.model.setup_caches(
                batch_size=1,
                dtype=self._dtype,
                encoder_max_seq_len=(
                    self.model_transform.image_seq_len if is_multimodal_input else None
                ),
                decoder_max_seq_len=total_response_length,
            )

        # 4. Pre-allocate causal mask and input_pos
        causal_mask = torch.tril(
            torch.ones(
                size=(total_response_length, total_response_length),
                dtype=torch.bool,
                device=self._device,
            )
        )
        input_pos = torch.arange(total_response_length)

        # 5. Collate to batch size of 1 and tensor-ify
        batch = {}
        if is_multimodal_input:
            batch = padded_collate_tiled_images_and_mask(
                [model_inputs], pad_direction="left", pad_max_images=1
            )
            batch["encoder_mask"] = batch["encoder_mask"][:, :seq_len]
            prompt = batch.pop("tokens").to(self._device)
        else:
            prompt = torch.tensor(
                model_inputs["tokens"], device=self._device
            ).unsqueeze(0)
        batch["mask"] = causal_mask[None, :seq_len]
        batch["input_pos"] = input_pos[None, :seq_len]
        utils.batch_to_device(batch, self._device)

        # 6. Prefill step
        generated_tokens = []
        t0 = time.perf_counter()
        logits = self.model(prompt, **batch)[:, -1]
        token = sample(logits, temperature=cfg.temperature, top_k=cfg.top_k)
        generated_tokens.append(token.item())

        if is_multimodal_input:
            # Don't need image info b/c we only support 1 image and it's been
            # processed by the model now
            batch.pop("encoder_input")
            batch["encoder_mask"] = batch["encoder_mask"][:, -1:]

        tracked_entropy = []
        tracked_varentropy = []
        # 7. Continue generating
        for i in range(cfg.max_new_tokens):
            # Update position and mask for incremental decoding
            batch["input_pos"] = input_pos[None, seq_len]
            batch["mask"] = causal_mask[None, seq_len, None, :]

            if token.item() in self.model_transform.stop_tokens:
                break

            logits = self.model(token, **batch)[:, -1]

            self._logger.debug(f"Logits: {logits}")
            entropy, varentropy = calculate_varentropy_logsoftmax(logits)
            tracked_entropy.append(entropy.item())
            tracked_varentropy.append(varentropy.item())
            self._logger.debug(f"Entropy: {entropy}, Varentropy: {varentropy}")

            token = sample(logits, temperature=0, top_k=10)
            generated_tokens.append(token.item())
            seq_len += 1
            if token.item() in [2186, 14682, 9388, 60]:
                generated_tokens = (
                    generated_tokens[:-1]
                    + self.model_transform.encode('",', add_bos=False, add_eos=False)
                    + self.model_transform.encode(
                        f'"ENTROPY": {max(tracked_entropy)}',
                        add_bos=False,
                        add_eos=False,
                    )
                    + self.model_transform.encode(
                        f',"VARENTROPY": {max(tracked_varentropy)}',
                        add_bos=False,
                        add_eos=False,
                    )
                    + self.model_transform.encode("},", add_bos=False, add_eos=False)
                )
                tracked_entropy = []
                tracked_varentropy = []

        t = time.perf_counter() - t0

        # 8. Translate tokens back to text
        decoded = self.model_transform.decode(generated_tokens)
        self._logger.info(f"\n\n{decoded}\n")

        # 9. Log metrics
        tokens_per_second = len(generated_tokens) / t
        self.log_metrics(total_time=t, tokens_per_second=tokens_per_second)

        # 10. Submit to DB
        self._submit_to_db(record_id, decoded)


import concurrent.futures


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)

    # Create four CUDA streams
    torch_stream1 = torch.cuda.Stream()
    torch_stream2 = torch.cuda.Stream()

    # Function to execute inference on stream1
    def run_inference_stream1():
        with torch.cuda.stream(torch_stream1):
            recipe = InferenceRecipe(cfg=cfg)
            recipe.setup(cfg=cfg)
            for _, (record_id, prompt) in enumerate(recipe.dataloader):
                recipe.generate(cfg=cfg, prompt=prompt, record_id=record_id)

    # Function to execute inference on stream2
    def run_inference_stream2():
        with torch.cuda.stream(torch_stream2):
            recipe = InferenceRecipe(cfg=cfg)
            recipe.setup(cfg=cfg)
            for _, (record_id, prompt) in enumerate(recipe.dataloader):
                recipe.generate(cfg=cfg, prompt=prompt, record_id=record_id)

    # Use ThreadPoolExecutor to run all four streams in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(run_inference_stream1)
        future2 = executor.submit(run_inference_stream2)

        # Wait for all tasks to complete
        concurrent.futures.wait([future1, future2])

    # Synchronize the CUDA streams after all work is done
    torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(main())
