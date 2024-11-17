import argparse
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path

import dspy
from dotenv import load_dotenv
from load_balancer_client import Client
from tqdm import tqdm
import logging
from src.data.llm_preprocess import preprocess_desc_data
from src.data.main import Data
from src.models import (
    CompatibliltyFinderV1,
    CompatibliltyFinderV2,
    CompatibliltyFinderV3,
)
from src.utils.helper import read_config
from src.utils.submission import (
    create_table,
    db_connect,
    insert_content_in_table,
    table_exists,
)

load_dotenv()


def process_chunk(cfg, record_id):
    # llm = dspy.OllamaLocal(model=cfg["ollama_model"], base_url=cfg["BASE_URL"])
    llm = dspy.LM(
        api_base=cfg["BASE_URL"],
        api_key="ollama",
        model=f"ollama/{cfg['ollama_model']}",
    )
    dspy.configure(lm=llm)

    data = Data(cfg)
    data.get(record_id=record_id)

    conn = db_connect(Path(cfg["DB_ROOT"]) / "submissions_bak.db")

    try:
        # try to get it right 3 times, if failed ignore
        desc_data_processed = preprocess_desc_data(
            ollama_host=cfg["BASE_URL"], desc_data=data.desc_data
        )["response"]
        result = dspy.Predict(CompatibliltyFinderV2, max_retries=3).forward(
            description=desc_data_processed,
            tags=data.tags_data,
            title=data.items_data,
        )
        logging.debug(
            f"Record ID:{record_id}\nInput Data:{desc_data_processed}\n{data.tags_data}\n{data.items_data}\n---\nOutput Data:{result.output.vehicles}\n"
        )
        logging.debug(result.rationale)
        for vehicle in result.output.vehicles:
            tmp = {
                "RECORD_ID": record_id,
                "FTMNT_YEAR": vehicle.year,
                "FTMNT_MAKE": vehicle.make,
                "FTMNT_MODEL": vehicle.model,
            }
            insert_content_in_table(conn, cfg["submission_table_name"], tmp)

        return True

    except ValueError:
        # logging.error(f"Record ID:{record_id} -- {result.output.vehicles}")
        logging.debug(
            f"--FAILED--\nRecord ID:{record_id}\nInput Data:{desc_data_processed}\n{data.tags_data}\n{data.items_data}\n---\nOutput Data:{result.output.vehicles}\n"
        )
        return False
    except Exception as e:
        logging.error(f"RECORD_ID:{record_id}", e)
        return False


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
