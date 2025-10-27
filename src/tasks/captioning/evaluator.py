import json
import os
from argparse import Namespace
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal

import numpy as np
from bert_score import score
from tqdm import tqdm
from transformers import logging

from src.models.base_model import BaseModel
from src.utils import prompt
from src.utils.common_utils import confirm_restart_if_exists, strip_trailing_slash
from src.utils.file_io import (
    convert_jsonl_to_json,
    load_captioning_data,
    load_completed_indices,
    load_data,
    load_distribution_data,
    save_json_data,
)
from src.utils.lfss_io import LabelReader, SkipReader

logging.set_verbosity_error()


def task(idx: str, model: BaseModel, lbl_meta_dir: str, vlm_captioning: str, args: Namespace) -> dict:
    r = SkipReader(lbl_meta_dir=lbl_meta_dir)
    skip = r.get(idx)
    if skip is not None:
        return None
    
    r = LabelReader(lbl_meta_dir=lbl_meta_dir)
    label = r.get(idx)
    if label is None:
        return None
    
    label = label.compact_json()
    
    case_en = None
    try:
        # translate the label to English
        if args.lfss_meta_type == "cn":
            p = prompt.translate_dent_json_zh2en.substitute(
                case=model.j2t(label)
                )
            case_en = model.generate_from_text(prompt=p)
        else:
            case_en = label
        
        # refine captioning (Refine all observed abnormalities from the captioning)
        p = prompt.captioning_extraction_intraoral_condition.substitute(
            case=model.j2t(vlm_captioning)
        )
        refine = model.generate_from_text(prompt=p, output_type=list)
        
        # generate the confusion matrix
        p = prompt.captioning_score_intraoral_condition.substitute(
            reference=model.j2t(case_en['items']),
            prediction=model.j2t(refine)
        )
        res = model.generate_from_text(prompt=p)
        
        return {
            "case_en": {idx: case_en},
            "refine": {idx: refine},
            "res": {idx: res}
        }
    
    except Exception as e:
        return {
            "case_en": {idx: case_en if case_en else {"failed": str(e)}},
            "refine": {idx: refine if refine else {"failed": str(e)}},
            "res": {idx: {"failed": str(e)}}
        }

def generate_captioning_confusion_matrix(model: BaseModel, yaml_cfg, args):
    data = load_data(os.path.join(args.project_root, args.save_root_dir, "prediction/captioning", args.model_name, "results.json"))
    data_keys = set(data.keys())
    
    completed = load_completed_indices(args)
    completed = confirm_restart_if_exists(args, completed)
    all_indices = [f"{i:09d}" for i in range(args.start, args.end + 1)]
    pending = [idx for idx in all_indices if idx not in completed and idx in data_keys]
    tqdm.write(f"Total tasks: {len(all_indices)} | Completed: {len(completed)} | Skipped: {len(all_indices) - len(completed) - len(pending)} | Pending: {len(pending)}")
    
    if not pending:
        print("All tasks already completed.")
    else:
        if args.lfss_meta_type == "cn":
            lbl_meta_dir=strip_trailing_slash(yaml_cfg["lfss"]["meta_cn_dir"])
        else:
            lbl_meta_dir=strip_trailing_slash(yaml_cfg["lfss"]["meta_en_dir"])
        
        with open(args.outfile, "a", encoding="utf-8") as f_out, \
            open(args.translatefile, "a", encoding="utf-8") as f_translate, \
            open(args.refinefile, "a", encoding="utf-8") as f_refine, \
            open(args.failfile, "a", encoding="utf-8") as f_fail:
            
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(task, idx, model, lbl_meta_dir, data[idx], args): idx for idx in pending}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="task", dynamic_ncols=True):
                    out = future.result()
                    if out:
                        case_en = out["case_en"]
                        refine = out["refine"]
                        result = out["res"]
                        if case_en:
                            if "failed" in case_en or (isinstance(case_en, dict)and "failed" in case_en[list(case_en.keys())[0]]):
                                f_fail.write(json.dumps(case_en, ensure_ascii=False) + "\n")
                                f_fail.flush()
                            else:
                                f_translate.write(json.dumps(case_en, ensure_ascii=False) + "\n")
                                f_translate.flush()
                        if refine:
                            if "failed" in refine or (isinstance(refine, dict) and "failed" in refine[list(refine.keys())[0]]):
                                f_fail.write(json.dumps(refine, ensure_ascii=False) + "\n")
                                f_fail.flush()
                            else:
                                f_refine.write(json.dumps(refine, ensure_ascii=False) + "\n")
                                f_refine.flush()
                        if result:
                            if "failed" in result or (isinstance(result, dict) and "failed" in result[list(result.keys())[0]]):
                                f_fail.write(json.dumps(result, ensure_ascii=False) + "\n")
                                f_fail.flush()
                            else:
                                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                                f_out.flush()
    
    convert_jsonl_to_json(args, jsonl_type="results")
    convert_jsonl_to_json(args, jsonl_type="translate")
    convert_jsonl_to_json(args, jsonl_type="failures")
    convert_jsonl_to_json(args, jsonl_type="refine")

def compute_confusion_matrix(confusion_dict):
    TP = confusion_dict.get("TP", 0)
    FP = confusion_dict.get("FP", 0)
    FN = confusion_dict.get("FN", 0)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "P": precision,
        "R": recall,
        "F1": f1
    }

def statistical_BERTScore(cands, refs, chunk=False, chunk_size=512):
    all_P, all_R, all_F1 = [], [], []

    if not chunk:
        while True:
            try:
                P, R, F1 = score(
                    cands,
                    refs,
                    lang="en",
                    verbose=True,
                    rescale_with_baseline=True
                )
                break
            except Exception as e:
                tqdm.write(e)
                tqdm.write(f"Retrying ...")
                continue
        return P, R, F1

    for i in tqdm(range(0, len(cands), chunk_size), desc="Scoring chunks", dynamic_ncols=True):
        batch_cands = cands[i:i+chunk_size]
        batch_refs = refs[i:i+chunk_size]
        P, R, F1 = score(
            batch_cands,
            batch_refs,
            lang="en",
            verbose=False,
            rescale_with_baseline=True
        )
        all_P.append(P)
        all_R.append(R)
        all_F1.append(F1)

    from torch import cat
    return cat(all_P), cat(all_R), cat(all_F1)

def calculated_all_matrice(model_json: dict, distribution: dict, save_dir: str, args: Namespace):
    model_summary = defaultdict(dict)
    for key, item in tqdm(model_json.items(), total=len(model_json), dynamic_ncols=True, desc=f"{args.model_name}"):
        for dataset_name, ids in distribution.items():
            if key in ids:
                if not model_summary[dataset_name]:
                    model_summary[dataset_name] = defaultdict(dict)
                if not model_summary[dataset_name][args.model_name]:
                    model_summary[dataset_name][args.model_name] = defaultdict(float)
                if not model_summary[args.model_name]:
                    model_summary[args.model_name] = defaultdict(float)
                
                # DS1, DS2, DS3
                model_summary[dataset_name][args.model_name]["total_P"] += item["P"]
                model_summary[dataset_name][args.model_name]["total_R"] += item["R"]
                model_summary[dataset_name][args.model_name]["total_F1"] += item["F1"]
                model_summary[dataset_name][args.model_name]["count"] += 1
                
                model_summary[dataset_name][args.model_name]["Avg_P"] =  model_summary[dataset_name][args.model_name]["total_P"] / model_summary[dataset_name][args.model_name]["count"]
                model_summary[dataset_name][args.model_name]["Avg_R"] =  model_summary[dataset_name][args.model_name]["total_R"] / model_summary[dataset_name][args.model_name]["count"]
                model_summary[dataset_name][args.model_name]["Avg_F1"] =  model_summary[dataset_name][args.model_name]["total_F1"] / model_summary[dataset_name][args.model_name]["count"]
            
                # MetaDent
                model_summary[args.model_name]["total_P"] += item["P"]
                model_summary[args.model_name]["total_R"] += item["R"]
                model_summary[args.model_name]["total_F1"] += item["F1"]
                model_summary[args.model_name]["count"] += 1
                
                model_summary[args.model_name]["Avg_P"] =  model_summary[args.model_name]["total_P"] / model_summary[args.model_name]["count"]
                model_summary[args.model_name]["Avg_R"] =  model_summary[args.model_name]["total_R"] / model_summary[args.model_name]["count"]
                model_summary[args.model_name]["Avg_F1"] =  model_summary[args.model_name]["total_F1"] / model_summary[args.model_name]["count"]
                
                # Keep three decimal places
                point = '0.000'
                model_summary[dataset_name][args.model_name]["Avg_P_decimal"] = float(Decimal(str(model_summary[dataset_name][args.model_name]["Avg_P"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                model_summary[dataset_name][args.model_name]["Avg_R_decimal"] = float(Decimal(str(model_summary[dataset_name][args.model_name]["Avg_R"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                model_summary[dataset_name][args.model_name]["Avg_F1_decimal"] = float(Decimal(str(model_summary[dataset_name][args.model_name]["Avg_F1"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                
                model_summary[args.model_name]["Avg_P_decimal"] = float(Decimal(str(model_summary[args.model_name]["Avg_P"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                model_summary[args.model_name]["Avg_R_decimal"] = float(Decimal(str(model_summary[args.model_name]["Avg_R"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                model_summary[args.model_name]["Avg_F1_decimal"] = float(Decimal(str(model_summary[args.model_name]["Avg_F1"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                
                break
    
    save_json_data(model_summary, save_dir, "results.json", title=f"{args.task} - {args.subtask} - {args.model_name}")

def run_captioning_evaluation(model: BaseModel, yaml_cfg, args):
    distribution = load_distribution_data()
    save_BertScore_dir = os.path.join(args.project_root, "metric", args.subtask, "BertScore", args.model_name)
    save_confusion_dir = os.path.join(args.project_root, "metric", args.subtask, "confusion_matrix", args.model_name)
    
    # 1. generate the BERTScore
    label_json = load_captioning_data()
    vlm_json_path = os.path.join(args.project_root, args.save_root_dir, "prediction/captioning", args.model_name, "results.json")
    if not os.path.exists(vlm_json_path):
        raise FileNotFoundError(f"{vlm_json_path} not found, please run predictor first.")
    with open(vlm_json_path, "r") as f:
        vlm_json = json.load(f)

    common_keys = [k for k in label_json if k in vlm_json]
    cands = [str(vlm_json[k]["description"]) for k in common_keys]
    refs = [str(label_json[k]["description"]) for k in common_keys]
    
    tqdm.write(f"Scoring {len(cands)} samples...")
    P, R, F1 = statistical_BERTScore(cands, refs, chunk=args.chunk, chunk_size=args.chunk_size)

    bert_score_json = {}
    for i, key in enumerate(common_keys):
        bert_score_json[key] = {
            "P": P[i].item(),
            "R": R[i].item(),
            "F1": F1[i].item()
        }
    save_json_data(bert_score_json, save_BertScore_dir, "per_sample.json", title=f"{args.task} - {args.subtask} - {args.model_name}")

    # 2. calculate the average score
    model_summary_BERTScore = {}

    for dataset_name, ids in distribution.items():
        Ps, Rs, F1s = [], [], []
        for _id in ids:
            if _id not in bert_score_json:
                continue
            sample = bert_score_json[_id]
            Ps.append(sample["P"])
            Rs.append(sample["R"])
            F1s.append(sample["F1"])

        if len(F1s) == 0:
            tqdm.write(f"No data for {args.model_name} - {dataset_name}")
            continue

        model_summary_BERTScore[dataset_name] = {
            "P": float(np.mean(Ps)),
            "R": float(np.mean(Rs)),
            "F1": float(np.mean(F1s)),
            "count": len(F1s)
        }

    all_P = [v["P"] for v in bert_score_json.values()]
    all_R = [v["R"] for v in bert_score_json.values()]
    all_F = [v["F1"] for v in bert_score_json.values()]
    model_summary_BERTScore["Overall"] = {
        "P": float(np.mean(all_P)),
        "R": float(np.mean(all_R)),
        "F1": float(np.mean(all_F)),
        "count": len(all_F)
    }
    save_json_data(model_summary_BERTScore, save_BertScore_dir, "per_sample_summary.json", title=f"{args.task} - {args.subtask} - {args.model_name}")
    
    # 3. generate the confusion matrix
    generate_captioning_confusion_matrix(model, yaml_cfg, args)
    
    merge_captioning_json = load_captioning_data()
    confusion_json_path = os.path.join(args.project_root, args.save_root_dir, args.task, args.subtask, args.model_name, "results.json")
    confusion_json = load_data(confusion_json_path)
    
    model_summary_confusion = defaultdict(dict)
    for key, value in tqdm(merge_captioning_json.items(), total=len(merge_captioning_json), dynamic_ncols=True, desc=f"{args.model_name}"):
        if key not in confusion_json:
            continue
        try:
            confusion = {
                "TP": confusion_json[key]["TP"],
                "FP": confusion_json[key]["FP"],
                "FN": confusion_json[key]["FN"],
                "TN": confusion_json[key]["TN"]
            }
            metrics = compute_confusion_matrix(confusion)
            confusion.update(metrics)
            model_summary_confusion[key] = confusion
        except Exception as e:
            tqdm.write(f"{key} - {e}")
            raise e
    
    save_json_data(model_summary_confusion, save_confusion_dir, "confusion_matrix.json", title=f"{args.task} - {args.subtask} - {args.model_name}")
    
    # 4. MetaDent, DS1, DS2, DS3
    calculated_all_matrice(bert_score_json, distribution, save_BertScore_dir, args)
    calculated_all_matrice(model_summary_confusion, distribution, save_confusion_dir, args)
