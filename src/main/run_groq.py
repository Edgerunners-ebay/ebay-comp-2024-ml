import argparse
import logging
import re
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import dspy
from dotenv import load_dotenv
from litellm.exceptions import (
    RateLimitError,
    ServiceUnavailableError,
    OpenAIError,
    BadRequestError,
)
from load_balancer_client import Client
from tqdm import tqdm

from src.data.llm_preprocess import preprocess_desc_data, preprocess_desc_data_groq
from src.data.main import Data
from src.models import (
    CompatibliltyFinderV1,
    CompatibliltyFinderV2,
    CompatibliltyFinderV3,
    ExtractYMM,
    ExtractYMMV2,
    ExtractYMMV3,
)
from src.utils.helper import read_config, process_output_v1, process_output_efficient
from src.utils.submission import (
    create_table,
    db_connect,
    insert_content_in_table,
    table_exists,
)

import litellm

litellm.set_verbose = True

load_dotenv(dotenv_path="/home/017534556/projects/ebay-comp-2024/.env")


def process_chunk(cfg, record_id):
    llm = dspy.LM(model=f"groq/{cfg['groq_model']}", max_tokens=8000, cache=False)
    dspy.configure(lm=llm)

    data = Data(cfg)
    data.get(record_id=record_id)

    conn = db_connect(Path(cfg["DB_ROOT"]) / "submissions_bak.db")

    counter = 2
    while counter > 0:  # try this for 2 times
        result = None
        try:
            # try to get it right 3 times, if failed ignore
            # desc_data_processed = preprocess_desc_data_groq(
            #     text=data.desc_data[:30000]
            # )  # limiting to 8192, 4chars * 8192=~32k
            desc_data_processed = data.desc_data
            result = dspy.ChainOfThought(ExtractYMMV3, max_retries=3).forward(
                description=desc_data_processed,
                tags=data.tags_data,
                title=data.items_data,
            )
            logging.debug(
                f"Record ID:{record_id}\nInput Data:{desc_data_processed}\n{data.tags_data}\n{data.items_data}\n---\nOutput Data:{result.output}\n"
            )
            output = result.output
            cleaned_output = re.sub(r"\.{2,}", "", output)

            # process_output_v1(conn, cfg, record_id, cleaned_output)

            process_output_efficient(conn, cfg, record_id, cleaned_output)

            return True

        except ServiceUnavailableError:
            logging.debug("Something messed up on Groq side, waiting for a minute")
            time.sleep(60)
            continue

        except RateLimitError:
            logging.debug("Hit the rate limit, waiting out a minute")
            time.sleep(60)
            continue

        except SyntaxError:
            logging.debug(
                "Something wrong with Eval, waiting a min and redoing it just in case"
            )
            time.sleep(60)
            counter -= 1
            continue

        except OpenAIError as e:
            logging.debug("Maybe failed to generate result?", e)
            counter -= 1
            continue

        except BadRequestError as e:
            logging.debug("Some bad request", e)
            counter -= 1
            continue

        except ValueError:
            logging.debug(
                f"--FAILED--\nRecord ID:{record_id}\nInput Data:{desc_data_processed}\n{data.tags_data}\n{data.items_data}\n---\nOutput Data:{result}\n"
            )
            return False

        except Exception as e:
            logging.error(f"RECORD_ID:{record_id}", e)
            return False
    logging.debug(f"RECORD_ID:{record_id} failed for some reason")


def worker(cfg):
    try:
        client = Client(server_ip="172.16.1.151", port=8000)
        client.register()
        while True:
            status_code, record_id_dict = client.post_task()
            record_id = record_id_dict.get("record_id")
            logging.debug(f"Record ID: {record_id}")
            if status_code == 410:
                break  # End process if no more tasks available
            _ = process_chunk(cfg, record_id)
            logging.debug(f"Record ID: {record_id} done")
            client.post_done()
    except Exception as e:
        logging.error(e)


def main(cfg, num_process=8):
    conn = db_connect(Path(cfg["DB_ROOT"]) / "submissions_bak.db")

    if not table_exists(conn, cfg["submission_table_name"]):
        create_table(conn, cfg["submission_table_name"])

    max_process = num_process

    pool = Pool(processes=max_process)
    print(max_process)
    for _ in range(max_process):
        pool.apply_async(worker, args=(cfg,))

    pool.close()  # No more tasks
    pool.join()  # wait for the tasks to finish
    print("All tasks are done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters to run pipeline")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to config"
    )
    parser.add_argument(
        "-n",
        "--num_process",
        type=int,
        required=False,
        help="Number of processes",
        default=8,
    )
    args = parser.parse_args()
    cfg = read_config(args.config)
    main(cfg, args.num_process)
