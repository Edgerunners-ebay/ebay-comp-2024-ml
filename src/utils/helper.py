import torch
import yaml
import json

from src.utils.submission import insert_content_in_table


def read_config(path_to_config):
    with open(path_to_config, "r") as f:
        return yaml.safe_load(f)


def get_split(phase):
    assert phase in ["train", "val", "test"], "Invalid phase"
    indices = list(range(5000))
    if phase == "train":
        return indices[:4000]
    elif phase == "val":
        return indices[4000:4500]
    else:
        return indices[4500:]


def convert_text_to_ymm_list(record_id, text):
    """This supports converting the model output decoded to English to Year-Make-Model format.

    Args:
        english (str): Model output tokens decoded to English.
    """
    ymms = text.split("<next>")
    outputs = []
    for ymm in ymms:
        split_text = ymm.split("<sep>")
        if len(split_text) != 3:
            continue
        year, make, model = ymm.split("<sep>")
        # print(f"Year: {year}, Make: {make}, Model: {model}")
        tmp = {
            "RECORD_ID": record_id,
            "FTMNT_YEAR": year,
            "FTMNT_MAKE": make,
            "FTMNT_MODEL": model,
        }
        outputs.append(tmp)
    return outputs


def convert_text_to_ymm_list_v2(record_id, text):
    """This supports converting the model output decoded to English to Year-Make-Model format.
    Doesn't hard-code the assistant seperator.
    Args:
        english (str): Model output tokens decoded to English.
    """
    ymms = text.split("<next>")
    # Remove the left-right splits of first <next> element as this goes in the prompt
    for _ in range(2):
        ymms.pop(0)
    outputs = []
    for idx, ymm in enumerate(ymms):
        split_text = ymm.split("<sep>")
        if len(split_text) != 3:
            continue
        year, make, model = ymm.split("<sep>")
        # print(f"Year: {year}, Make: {make}, Model: {model}")
        if idx == 0:
            year = year[-4:]
        tmp = {
            "RECORD_ID": record_id,
            "FTMNT_YEAR": year,
            "FTMNT_MAKE": make,
            "FTMNT_MODEL": model,
        }
        outputs.append(tmp)
    return outputs


def calculate_varentropy_logsoftmax(logits, axis: int = -1):
    LN_2 = 0.69314718056
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = torch.log_softmax(logits, axis=axis)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, axis=axis) / LN_2  # Convert to base-2
    varentropy = torch.sum(
        probs * (log_probs / LN_2 + entropy[..., None]) ** 2, axis=axis
    )
    return entropy, varentropy


def process_output_v1(conn, cfg, record_id, output):
    for vehicle in eval(output):
        tmp = {
            "RECORD_ID": record_id,
            "FTMNT_YEAR": vehicle["year"],
            "FTMNT_MAKE": vehicle["make"],
            "FTMNT_MODEL": vehicle["model"],
        }
        insert_content_in_table(conn, cfg["submission_table_name"], tmp)


def process_output_efficient(conn, cfg, record_id, output):
    ymms = []
    if isinstance(output, str):
        json_output = json.loads(output)
    else:
        json_output = output
    for make, models in json_output.items():
        for model, years in models.items():
            for year in years:
                ymms.append(
                    {
                        "RECORD_ID": record_id[0]
                        if isinstance(record_id, list)
                        else record_id,
                        "FTMNT_YEAR": year,
                        "FTMNT_MAKE": make,
                        "FTMNT_MODEL": model,
                    }
                )
    for ymm in ymms:
        insert_content_in_table(conn, cfg["submission_table_name"], ymm)
