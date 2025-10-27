import json
import os
from argparse import Namespace
from typing import Any, Dict, Literal

import pandas as pd
from tqdm import tqdm

from src.utils.common_utils import *

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
data_dir = os.path.join(project_root, "data")

def load_distribution_data(data_path=os.path.join(data_dir, "distribution.json")) -> Dict[str, Any]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return json.load(open(data_path, "r", encoding="utf-8"))

def load_vqa_data(data_path=os.path.join(data_dir, "vqa.json")) -> Dict[str, Any]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return json.load(open(data_path, "r", encoding="utf-8"))

def load_classification_data(data_path=os.path.join(data_dir, "classification.json")) -> Dict[str, Any]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return json.load(open(data_path, "r", encoding="utf-8"))

def load_captioning_data(data_path=os.path.join(data_dir, "captioning.json")) -> Dict[str, Any]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return json.load(open(data_path, "r", encoding="utf-8"))

def load_data(data_path: str) -> Dict[str, Any]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return json.load(open(data_path, "r", encoding="utf-8"))

def save_json_data(data: Dict[str, Any], save_dir: str, save_file_name: str, title: str = "Saved") -> None:
    save_path = os.path.join(strip_trailing_slash(save_dir), save_file_name)
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        json.dump(data, open(save_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        tqdm.write(f"[{title}] Saved data to '{save_path}'")
    except Exception as e:
        tqdm.write(f"[{title}] Failed to save data to '{save_path}': {e}")

def save_csv_data(data: pd.DataFrame, save_dir: str, save_file_name: str, title: str = "Saved") -> None:
    save_path = os.path.join(strip_trailing_slash(save_dir), save_file_name)
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path, index=False)
        tqdm.write(f"[{title}] Saved data to '{save_path}'")
    except Exception as e:
        tqdm.write(f"[{title}] Failed to save data to '{save_path}': {e}")

def load_completed_indices(args: Namespace) -> set:
    completed = set()
    if os.path.exists(args.outfile):
        with open(args.outfile, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    completed.update(data.keys())
                except Exception:
                    continue
    return completed

def convert_jsonl_to_json(args: Namespace, jsonl_type: Literal["results", "translate", "failures", "refine"] = "results"):
    results = {}
    
    if jsonl_type == "results":
        if os.path.exists(args.outfile):
            with open(args.outfile, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        results.update(data)
                    except Exception:
                        continue
    elif jsonl_type == "translate":
        if os.path.exists(args.translatefile):
            with open(args.translatefile, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        results.update(data)
                    except Exception:
                        continue
    elif jsonl_type == "failures":
        if os.path.exists(args.failfile):
            with open(args.failfile, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        results.update(data)
                    except Exception:
                        continue
    elif jsonl_type == "refine":
        if os.path.exists(args.refinefile):
            with open(args.refinefile, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        results.update(data)
                    except Exception:
                        continue

    sorted_results = {k: results[k] for k in sorted(results.keys())}
    
    if jsonl_type == "results":
        with open(change_path_suffix(args.outfile, ".json"), "w", encoding="utf-8") as f:
            json.dump(sorted_results, f, ensure_ascii=False, indent=2)
        tqdm.write(f"[{args.task} - {args.subtask} - {args.model_name}] Saved data to '{change_path_suffix(args.outfile, '.json')}'")
    
    elif jsonl_type == "translate":
        with open(change_path_suffix(args.translatefile, ".json"), "w", encoding="utf-8") as f:
            json.dump(sorted_results, f, ensure_ascii=False, indent=2)
        tqdm.write(f"[{args.task} - {args.subtask} - {args.model_name}] Saved data to '{change_path_suffix(args.translatefile, '.json')}'")
    
    elif jsonl_type == "failures":
        with open(change_path_suffix(args.failfile, ".json"), "w", encoding="utf-8") as f:
            json.dump(sorted_results, f, ensure_ascii=False, indent=2)
        tqdm.write(f"[{args.task} - {args.subtask} - {args.model_name}] Saved data to '{change_path_suffix(args.failfile, '.json')}'")
        if sorted_results:
            tqdm.write(f"\033[91m[WARNING] [{args.task} - {args.subtask} - {args.model_name}] Failed tasks: {len(sorted_results)}\033[0m")

    elif jsonl_type == "refine":
        with open(change_path_suffix(args.refinefile, ".json"), "w", encoding="utf-8") as f:
            json.dump(sorted_results, f, ensure_ascii=False, indent=2)
        tqdm.write(f"[{args.task} - {args.subtask} - {args.model_name}] Saved data to '{change_path_suffix(args.refinefile, '.json')}'")
