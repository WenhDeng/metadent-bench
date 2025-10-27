import os
from argparse import Namespace
from collections import defaultdict
from decimal import Decimal

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm

from src.utils.file_io import (
    load_classification_data,
    load_data,
    load_distribution_data,
    save_csv_data,
    save_json_data,
)


def compute_confusion_matrix(reference, prediction, all_labels=None):
    try:
        ref_set = set(reference)
        pred_set = set(prediction)
    except Exception as e:
        print(f"Error: {e}")
        breakpoint()
    
    # Exact Match
    exact_match = 1 if ref_set == pred_set else 0

    tp = len(ref_set & pred_set)
    fn = len(ref_set - pred_set)
    fp = len(pred_set - ref_set)
    
    result = {'C1': None, 'C2': None, 'C3': None, 'C4': None, 'C5': None, 'C6': None, 'C7': None, 'C8': None, 'C9': None, 'C10': None, 'C11': None, 'C12': None, 'C13': None, 'C14': None, 'C15': None, 'C16': None, 'C17': None, 'C18': None}
    result.update({k: 0 for k in ref_set - pred_set})
    result.update({k: 1 for k in ref_set & pred_set})
    
    if all_labels is not None:
        all_set = set(all_labels)
        tn = len(all_set - (ref_set | pred_set))
    else:
        tn = None
    
    if not ref_set and not pred_set:
        tp = 1
    
    result.update({"TP": tp, "FN": fn, "FP": fp, "TN": tn, "Exact_Match": exact_match})
    return result


def compute_metrics(confusion_dict):
    TP = confusion_dict.get("TP", 0)
    FP = confusion_dict.get("FP", 0)
    FN = confusion_dict.get("FN", 0)
    # TN = confusion_dict.get("TN", None)
    exact_match = confusion_dict.get("Exact_Match", 0)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "P": precision,
        "R": recall,
        "F1": f1,
        "Exact_Match": exact_match
    }

def calculate_all_metrics(model_json: dict, distribution: dict, save_dir: str, args: Namespace):
    model_summary = defaultdict(dict)
    classification_categories = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18']
    for key, item in tqdm(model_json.items(), total=len(model_json), dynamic_ncols=True, desc=f"{args.model_name}"):
        for dataset_name, ids in distribution.items():
            if key in ids:
                if not model_summary[dataset_name]:
                    model_summary[dataset_name] = defaultdict(dict)
                if not model_summary[dataset_name][args.model_name]:
                    model_summary[dataset_name][args.model_name] = defaultdict(float)
                    for category in classification_categories:
                        model_summary[dataset_name][args.model_name][category] = 0
                        model_summary[dataset_name][args.model_name][f"{category}_cnt"] = 0
                        model_summary[dataset_name][args.model_name][f"Avg_{category}"] = 0
                    model_summary[dataset_name][args.model_name][f"total_C_cnt"] = 0
                if not model_summary[args.model_name]:
                    model_summary[args.model_name] = defaultdict(float)
                    for category in classification_categories:
                        model_summary[args.model_name][category] = 0
                        model_summary[args.model_name][f"{category}_cnt"] = 0
                        model_summary[args.model_name][f"Avg_{category}"] = 0
                    model_summary[args.model_name][f"total_C_cnt"] = 0
                
                # DS1, DS2, DS3
                for category in classification_categories:
                    if item[category] is None:
                        continue
                    model_summary[dataset_name][args.model_name][category] += item[category]
                    model_summary[dataset_name][args.model_name][f"{category}_cnt"] += 1
                    model_summary[dataset_name][args.model_name][f"total_C_cnt"] += 1
                    model_summary[dataset_name][args.model_name][f"Avg_{category}"] = model_summary[dataset_name][args.model_name][category] / model_summary[dataset_name][args.model_name][f"{category}_cnt"]
                model_summary[dataset_name][args.model_name]["total_P"] += item["P"]
                model_summary[dataset_name][args.model_name]["total_R"] += item["R"]
                model_summary[dataset_name][args.model_name]["total_F1"] += item["F1"]
                model_summary[dataset_name][args.model_name]["total_Exact_Match"] += item["Exact_Match"]
                model_summary[dataset_name][args.model_name]["count"] += 1
                
                model_summary[dataset_name][args.model_name]["Avg_P"] =  model_summary[dataset_name][args.model_name]["total_P"] / model_summary[dataset_name][args.model_name]["count"]
                model_summary[dataset_name][args.model_name]["Avg_R"] =  model_summary[dataset_name][args.model_name]["total_R"] / model_summary[dataset_name][args.model_name]["count"]
                model_summary[dataset_name][args.model_name]["Avg_F1"] =  model_summary[dataset_name][args.model_name]["total_F1"] / model_summary[dataset_name][args.model_name]["count"]
                model_summary[dataset_name][args.model_name]["Avg_Exact_Match"] =  model_summary[dataset_name][args.model_name]["total_Exact_Match"] / model_summary[dataset_name][args.model_name]["count"]
            
                # MetaDent
                for category in classification_categories:
                    if item[category] is None:
                        continue
                    model_summary[args.model_name][category] += item[category]
                    model_summary[args.model_name][f"{category}_cnt"] += 1
                    model_summary[args.model_name][f"total_C_cnt"] += 1
                    model_summary[args.model_name][f"Avg_{category}"] = model_summary[args.model_name][category] / model_summary[args.model_name][f"{category}_cnt"]
                model_summary[args.model_name]["total_P"] += item["P"]
                model_summary[args.model_name]["total_R"] += item["R"]
                model_summary[args.model_name]["total_F1"] += item["F1"]
                model_summary[args.model_name]["total_Exact_Match"] += item["Exact_Match"]
                model_summary[args.model_name]["count"] += 1
                
                model_summary[args.model_name]["Avg_P"] =  model_summary[args.model_name]["total_P"] / model_summary[args.model_name]["count"]
                model_summary[args.model_name]["Avg_R"] =  model_summary[args.model_name]["total_R"] / model_summary[args.model_name]["count"]
                model_summary[args.model_name]["Avg_F1"] =  model_summary[args.model_name]["total_F1"] / model_summary[args.model_name]["count"]
                model_summary[args.model_name]["Avg_Exact_Match"] =  model_summary[args.model_name]["total_Exact_Match"] / model_summary[args.model_name]["count"]
                
                # Keep three decimal places
                point = '0.000'
                model_summary[dataset_name][args.model_name]["Avg_P_decimal"] = float(Decimal(str(model_summary[dataset_name][args.model_name]["Avg_P"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                model_summary[dataset_name][args.model_name]["Avg_R_decimal"] = float(Decimal(str(model_summary[dataset_name][args.model_name]["Avg_R"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                model_summary[dataset_name][args.model_name]["Avg_F1_decimal"] = float(Decimal(str(model_summary[dataset_name][args.model_name]["Avg_F1"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                model_summary[dataset_name][args.model_name]["Avg_Exact_Match_decimal"] = float(Decimal(str(model_summary[dataset_name][args.model_name]["Avg_Exact_Match"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                
                model_summary[args.model_name]["Avg_P_decimal"] = float(Decimal(str(model_summary[args.model_name]["Avg_P"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                model_summary[args.model_name]["Avg_R_decimal"] = float(Decimal(str(model_summary[args.model_name]["Avg_R"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                model_summary[args.model_name]["Avg_F1_decimal"] = float(Decimal(str(model_summary[args.model_name]["Avg_F1"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                model_summary[args.model_name]["Avg_Exact_Match_decimal"] = float(Decimal(str(model_summary[args.model_name]["Avg_Exact_Match"])).quantize(Decimal(point), rounding="ROUND_HALF_UP"))
                
                break
    
    save_json_data(model_summary, save_dir, "results.json", title=f"{args.task} - {args.subtask} - {args.model_name}")


def run_classification_evaluation(args):
    distribution_json = load_distribution_data()
    label_json = load_classification_data()
    # 18-class
    labels = set(["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18"])
    
    # df_not_exact_match = pd.DataFrame(columns=["ID", "Label", "Prediction"])
    # df_exact_match = pd.DataFrame(columns=["ID", "Label", "Prediction"])
    df_per_class_acc = pd.DataFrame(columns=["ID", "Label", "Prediction", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18"])
    data_path = os.path.join(args.project_root, args.save_root_dir, "prediction", args.subtask, args.model_name, "results.json")
    model_json = load_data(data_path)
    
    per_sample = defaultdict(dict)
    columns = ["ID"] + [f"C{i}" for i in range(1, 19)]
    rows = []
    for id_, labels in tqdm(label_json.items(), total=len(label_json), dynamic_ncols=True, desc=f"{args.model_name}"):
        row = {"ID": id_}
        for i in range(1, 19):
            label = f"C{i}"
            row[label] = 1 if label in labels else 0
        rows.append(row)
    df = pd.DataFrame(rows, columns=columns)
    rows = []
    gt = df.copy()
    
    for key, value in tqdm(label_json.items(), total=len(label_json), dynamic_ncols=True, desc=f"{args.model_name}"):
        if key not in model_json:
            continue
        
        ids = set()
        for item in model_json[key]:
            try:
                if not item or isinstance(item, str):
                    continue
                if isinstance(item, list):
                    for tmp in item:
                        if "id" in tmp:
                            ids.add(tmp["id"])
                        elif "1" in tmp:
                            ids.add(tmp["1"])
                        else:
                            raise Exception("")
                else:
                    if "id" in item:
                        ids.add(item["id"])
                    elif "1" in item:
                        ids.add(item["1"])
                    else:
                        raise Exception("")
            except Exception as e:
                # There are other error formats, directly through string matching.
                s = str(model_json[key])
                for l in labels:
                    if l in s:
                        ids.add(l)
                break
        confusion = compute_confusion_matrix(value, list(ids))
        row = {"ID": key}
        for i in range(1, 19):
            label = f"C{i}"
            row[label] = 1 if label in ids else 0
        rows.append(row)
        
        df_per_class_acc = pd.concat([
            df_per_class_acc,
            pd.DataFrame([{
                "ID": str(key), 
                "Label": "#".join(value) if value else "None", 
                "Prediction": "#".join(list(ids)) if list(ids) else "None", 
                "C1": confusion["C1"] if confusion["C1"] is not None else 1, 
                "C2": confusion["C2"] if confusion["C2"] is not None else 1, 
                "C3": confusion["C3"] if confusion["C3"] is not None else 1, 
                "C4": confusion["C4"] if confusion["C4"] is not None else 1, 
                "C5": confusion["C5"] if confusion["C5"] is not None else 1, 
                "C6": confusion["C6"] if confusion["C6"] is not None else 1, 
                "C7": confusion["C7"] if confusion["C7"] is not None else 1, 
                "C8": confusion["C8"] if confusion["C8"] is not None else 1, 
                "C9": confusion["C9"] if confusion["C9"] is not None else 1, 
                "C10": confusion["C10"] if confusion["C10"] is not None else 1, 
                "C11": confusion["C11"] if confusion["C11"] is not None else 1, 
                "C12": confusion["C12"] if confusion["C12"] is not None else 1, 
                "C13": confusion["C13"] if confusion["C13"] is not None else 1, 
                "C14": confusion["C14"] if confusion["C14"] is not None else 1, 
                "C15": confusion["C15"] if confusion["C15"] is not None else 1, 
                "C16": confusion["C16"] if confusion["C16"] is not None else 1, 
                "C17": confusion["C17"] if confusion["C17"] is not None else 1, 
                "C18": confusion["C18"] if confusion["C18"] is not None else 1
            }])
        ], ignore_index=True)
        
        # # Record the pictures that are not completely matched
        # if key in distribution_json["DS2"]:
        #     if confusion["Exact_Match"] != 1:
        #         df_not_exact_match = pd.concat([
        #             df_not_exact_match,
        #             pd.DataFrame([{"ID": str(key), "Label": "#".join(value) if value else "None", "Prediction": "#".join(list(ids)) if list(ids) else "None"}])
        #         ], ignore_index=True)
        #     else:
        #         df_exact_match = pd.concat([
        #             df_exact_match,
        #             pd.DataFrame([{"ID": str(key), "Label": "#".join(value) if value else "None", "Prediction": "#".join(list(ids)) if list(ids) else "None"}])
        #         ], ignore_index=True)
        
        metrics = compute_metrics(confusion)
        confusion.update(metrics)
        per_sample[key] = confusion
    
    save_result_dir = os.path.join(args.project_root, "metric", args.subtask, "Exact_Match", args.model_name)
    save_json_data(per_sample, save_result_dir, "per_sample.json", title=f"{args.task} - {args.subtask} - {args.model_name}")
    
    save_dir = os.path.join(args.project_root, "metric", args.subtask, "Exact_Match", args.model_name)
    
    # [Classification] P, R, F1
    pred = pd.DataFrame(rows, columns=columns)
    save_csv_data(pred, save_dir, "results.csv", title=f"{args.task} - {args.subtask} - {args.model_name}")
    common_ids = set(gt["ID"]) & set(pred["ID"])
    gt = gt[gt["ID"].isin(common_ids)].set_index("ID")
    pred = pred[pred["ID"].isin(common_ids)].set_index("ID")
    cols = [c for c in gt.columns if c.startswith("C")]
    results = []
    for c in cols:
        y_true = gt[c]
        y_pred = pred[c]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        results.append({
            "Class": c,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })
    df_result = pd.DataFrame(results)
    macro_P = df_result["Precision"].mean()
    macro_R = df_result["Recall"].mean()
    macro_F1 = df_result["F1"].mean()
    micro_P = precision_score(gt[cols].values.flatten(), pred[cols].values.flatten(), zero_division=0)
    micro_R = recall_score(gt[cols].values.flatten(), pred[cols].values.flatten(), zero_division=0)
    micro_F1 = f1_score(gt[cols].values.flatten(), pred[cols].values.flatten(), zero_division=0)
    summary = pd.DataFrame({
        "Metric": ["Macro", "Micro"],
        "Precision": [macro_P, micro_P],
        "Recall": [macro_R, micro_R],
        "F1": [macro_F1, micro_F1]
    })
    save_csv_data(df_result, save_dir, "classwise_metrics.csv", title=f"{args.task} - {args.subtask} - {args.model_name}")
    save_csv_data(summary, save_dir, "overall_metrics.csv", title=f"{args.task} - {args.subtask} - {args.model_name}")
    
    # Calculate the accuracy of each category
    df_per_class_acc = pd.concat([
        df_per_class_acc,
        pd.DataFrame([{
            "ID": "Accuracy", 
            "Label": "Accuracy", 
            "Prediction": "Accuracy", 
            "C1": df_per_class_acc["C1"].mean(), 
            "C2": df_per_class_acc["C2"].mean(), 
            "C3": df_per_class_acc["C3"].mean(), 
            "C4": df_per_class_acc["C4"].mean(), 
            "C5": df_per_class_acc["C5"].mean(), 
            "C6": df_per_class_acc["C6"].mean(), 
            "C7": df_per_class_acc["C7"].mean(), 
            "C8": df_per_class_acc["C8"].mean(), 
            "C9": df_per_class_acc["C9"].mean(), 
            "C10": df_per_class_acc["C10"].mean(), 
            "C11": df_per_class_acc["C11"].mean(), 
            "C12": df_per_class_acc["C12"].mean(), 
            "C13": df_per_class_acc["C13"].mean(), 
            "C14": df_per_class_acc["C14"].mean(), 
            "C15": df_per_class_acc["C15"].mean(), 
            "C16": df_per_class_acc["C16"].mean(), 
            "C17": df_per_class_acc["C17"].mean(), 
            "C18": df_per_class_acc["C18"].mean()
        }])
    ])
    save_csv_data(df_per_class_acc, save_dir, "per_class_acc.csv", title=f"{args.task} - {args.subtask} - {args.model_name}")
    
    calculate_all_metrics(per_sample, distribution_json, save_dir, args)
