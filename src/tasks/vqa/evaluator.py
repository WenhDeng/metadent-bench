import os
from argparse import Namespace
from collections import defaultdict
from decimal import Decimal

from tqdm import tqdm

from src.utils.file_io import (
    load_data,
    load_distribution_data,
    load_vqa_data,
    save_json_data,
)


def calculate_all_metrics(model_json: dict, distribution: dict, save_dir: str, args: Namespace):
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
                model_summary[dataset_name][args.model_name]["multiple_choice_acc_count"] += item["multiple_choice_acc_count"]
                model_summary[dataset_name][args.model_name]["multiple_choice_count"] += item["multiple_choice_count"]
                model_summary[dataset_name][args.model_name]["multiple_choice_acc"] = model_summary[dataset_name][args.model_name]["multiple_choice_acc_count"] / model_summary[dataset_name][args.model_name]["multiple_choice_count"] if model_summary[dataset_name][args.model_name]["multiple_choice_count"] else 1
                
                model_summary[dataset_name][args.model_name]["judge_acc_count"] += item["judge_acc_count"]
                model_summary[dataset_name][args.model_name]["judge_count"] += item["judge_count"]
                model_summary[dataset_name][args.model_name]["judge_acc"] = model_summary[dataset_name][args.model_name]["judge_acc_count"] / model_summary[dataset_name][args.model_name]["judge_count"] if model_summary[dataset_name][args.model_name]["judge_count"] else 1
                
                model_summary[dataset_name][args.model_name]["total_acc"] = (model_summary[dataset_name][args.model_name]["multiple_choice_acc_count"] + model_summary[dataset_name][args.model_name]["judge_acc_count"]) / (model_summary[dataset_name][args.model_name]["multiple_choice_count"] + model_summary[dataset_name][args.model_name]["judge_count"])
                
                # MetaDent
                model_summary[args.model_name]["multiple_choice_acc_count"] += item["multiple_choice_acc_count"]
                model_summary[args.model_name]["multiple_choice_count"] += item["multiple_choice_count"]
                model_summary[args.model_name]["multiple_choice_acc"] = model_summary[args.model_name]["multiple_choice_acc_count"] / model_summary[args.model_name]["multiple_choice_count"] if model_summary[args.model_name]["multiple_choice_count"] else 1
                
                model_summary[args.model_name]["judge_acc_count"] += item["judge_acc_count"]
                model_summary[args.model_name]["judge_count"] += item["judge_count"]
                model_summary[args.model_name]["judge_acc"] = model_summary[args.model_name]["judge_acc_count"] / model_summary[args.model_name]["judge_count"] if model_summary[args.model_name]["judge_count"] else 1
                
                model_summary[args.model_name]["total_acc"] = (model_summary[args.model_name]["multiple_choice_acc_count"] + model_summary[args.model_name]["judge_acc_count"]) / (model_summary[args.model_name]["multiple_choice_count"] + model_summary[args.model_name]["judge_count"])
                
                # Keep three decimal places
                point = '0.000'
                model_summary[dataset_name][args.model_name]["multiple_choice_acc_decimal"] = float(Decimal(str(model_summary[dataset_name][args.model_name]["multiple_choice_acc"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                model_summary[dataset_name][args.model_name]["judge_acc_decimal"] = float(Decimal(str(model_summary[dataset_name][args.model_name]["judge_acc"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                model_summary[dataset_name][args.model_name]["total_acc_decimal"] = float(Decimal(str(model_summary[dataset_name][args.model_name]["total_acc"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                
                model_summary[args.model_name]["multiple_choice_acc_decimal"] = float(Decimal(str(model_summary[args.model_name]["multiple_choice_acc"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                model_summary[args.model_name]["judge_acc_decimal"] = float(Decimal(str(model_summary[args.model_name]["judge_acc"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                model_summary[args.model_name]["total_acc_decimal"] = float(Decimal(str(model_summary[args.model_name]["total_acc"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                
                break
    
    save_json_data(model_summary, save_dir, "results.json", title=f"{args.task} - {args.subtask} - {args.model_name}")


def run_vqa_evaluation(args):
    distribution_json = load_distribution_data()
    label_json = load_vqa_data()
    data_path = os.path.join(args.project_root, args.save_root_dir, "prediction", args.subtask, args.model_name, "results.json")
    model_json = load_data(data_path)
    
    per_sample = defaultdict(dict)
    for key, value in tqdm(label_json.items(), total=len(label_json), dynamic_ncols=True, desc=f"{args.model_name}"):
        if key not in model_json:
            continue
        if not per_sample[key]:
            per_sample[key] = defaultdict(float)
            per_sample[key]["multiple_choice_acc_count"] = 0
            per_sample[key]["multiple_choice_count"] = 0
            per_sample[key]["multiple_choice_acc"] = 1
            per_sample[key]["judge_acc_count"] = 0
            per_sample[key]["judge_count"] = 0
            per_sample[key]["judge_acc"] = 1
            per_sample[key]["total_acc"] = 1
        for item in model_json[key]:
            if not item["AI_answer"]:
                continue
            
            if item["question_type"] == "multiple_choice":
                per_sample[key]["multiple_choice_count"] += 1
                if item["answer"] == item["AI_answer"]:
                    per_sample[key]["multiple_choice_acc_count"] += 1
            else:
                per_sample[key]["judge_count"] += 1
                if item["answer"] == item["AI_answer"]:
                    per_sample[key]["judge_acc_count"] += 1
        
        if per_sample[key]["multiple_choice_count"]:
            per_sample[key]["multiple_choice_acc"] = per_sample[key]["multiple_choice_acc_count"] / per_sample[key]["multiple_choice_count"]
        if per_sample[key]["judge_count"]:
            per_sample[key]["judge_acc"] = per_sample[key]["judge_acc_count"] / per_sample[key]["judge_count"]
        if per_sample[key]["multiple_choice_count"] + per_sample[key]["judge_count"] > 0:
            per_sample[key]["total_acc"] = (per_sample[key]["multiple_choice_acc_count"] + per_sample[key]["judge_acc_count"]) / (per_sample[key]["multiple_choice_count"] + per_sample[key]["judge_count"])
    
    save_result_dir = os.path.join(args.project_root, "metric", args.subtask, "Accuracy", args.model_name)
    save_json_data(per_sample, save_result_dir, "per_sample.json", title=f"{args.task} - {args.subtask} - {args.model_name}")
    
    calculate_all_metrics(per_sample, distribution_json, save_result_dir, args)
