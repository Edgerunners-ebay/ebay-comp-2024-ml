from .submission import (
    db_connect,
    read_table,
    read_record_ids,
    read_table_w_entropy_varentropy,
)
from .helper import get_split
from tqdm import tqdm


def pre_process(result_tuple):
    record_id, year, make, model = result_tuple
    make = "".join(e for e in make.lower() if e.isalnum())
    model = "".join(e for e in model.lower() if e.isalnum())
    return (record_id, year, make, model)


def pre_process_w_entropy(result_tuple):
    record_id, year, make, model, entropy, varentropy = result_tuple
    make = "".join(e for e in make.lower() if e.isalnum())
    model = "".join(e for e in model.lower() if e.isalnum())
    return (record_id, year, make, model, float(entropy), float(varentropy))


def eval_fbeta(pred_table, gt_table, beta=0.2, phase=None, entropy_threshold=None):
    conn = db_connect("/home/017534556/projects/ebay-comp-2024/db/submissions_bak.db")
    if not phase:
        ids_to_query = read_record_ids(conn, pred_table)
    else:
        ids_to_query = get_split(phase)
    tp, fp, fn = 0, 0, 0
    for _id in tqdm(ids_to_query, total=len(ids_to_query), desc="Evaluating"):
        content = read_table_w_entropy_varentropy(
            conn, pred_table, f"RECORD_ID IS {_id}"
        )
        pred_results = set([pre_process_w_entropy(x) for x in content])
        content = read_table(conn, gt_table, f"RECORD_ID IS {_id}")
        gt_results = [pre_process(x) for x in content]
        for pred in pred_results:
            if pred[-2] > entropy_threshold if entropy_threshold else 0.9:
                continue
            if pred[:-2] in gt_results:
                tp += 1
            else:
                fp += 1
        pred_results = [x[:-2] for x in pred_results]
        for gt in gt_results:
            if gt not in pred_results:
                fn += 1
    print(f"tp: {tp}, fp: {fp}, fn: {fn}")
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

    print(f"Precision: {precision}, Recall: {recall}, F0.2: {fbeta}")
    conn.close()
    return fbeta


def eval_fbeta_zeroshot(pred_table, gt_table, beta=0.2, phase=None):
    conn = db_connect("/home/017534556/projects/ebay-comp-2024/db/submissions_bak.db")
    if not phase:
        ids_to_query = read_record_ids(conn, pred_table)
    else:
        ids_to_query = get_split(phase)
    tp, fp, fn = 0, 0, 0
    for _id in tqdm(ids_to_query, total=len(ids_to_query), desc="Evaluating"):
        content = read_table(conn, pred_table, f"RECORD_ID IS {_id}")
        pred_results = set([pre_process(x) for x in content])
        content = read_table(conn, gt_table, f"RECORD_ID IS {_id}")
        gt_results = [pre_process(x) for x in content]
        for pred in pred_results:
            if pred in gt_results:
                tp += 1
            else:
                fp += 1
        for gt in gt_results:
            if gt not in pred_results:
                fn += 1
    print(f"tp: {tp}, fp: {fp}, fn: {fn}")
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

    print(f"Precision: {precision}, Recall: {recall}, F0.2: {fbeta}")
    conn.close()
    return fbeta
