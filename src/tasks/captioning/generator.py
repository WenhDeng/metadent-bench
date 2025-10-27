import json
import os
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.models.base_model import BaseModel
from src.utils import prompt
from src.utils.common_utils import confirm_restart_if_exists, strip_trailing_slash
from src.utils.file_io import convert_jsonl_to_json, load_completed_indices
from src.utils.lfss_io import LabelReader, SkipReader


def task(idx: str, model: BaseModel, lbl_meta_dir: str, args: Namespace) -> dict:
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
        if args.lfss_meta_type == "cn":
            p = prompt.translate_dent_json_zh2en.substitute(
                case=model.j2t(label)
                )
            case_en = model.generate_from_text(prompt=p)
        else:
            case_en = label
        
        p = prompt.summary_intraoral_condition.substitute(
            case=model.j2t(case_en)
        )
        res = model.generate_from_text(prompt=p, temperature=0.6)
        
        return {
            "case_en": {idx: case_en},
            "res": {idx: res}
        }
    
    except Exception as e:
        return {
            "case_en": {idx: case_en if case_en else {"failed": str(e)}},
            "res": {idx: {"failed": str(e)}}
        }

def run_captioning_generation(model: BaseModel, yaml_cfg, args):
    completed = load_completed_indices(args)
    completed = confirm_restart_if_exists(args, completed)
    all_indices = [f"{i:09d}" for i in range(args.start, args.end + 1)]
    pending = [idx for idx in all_indices if idx not in completed]
    tqdm.write(f"Total tasks: {len(all_indices)} | Completed: {len(completed)} | Pending: {len(pending)}")
    
    if not pending:
        print("All tasks already completed.")
    else:
        if args.lfss_meta_type == "cn":
            lbl_meta_dir=strip_trailing_slash(yaml_cfg["lfss"]["meta_cn_dir"])
        else:
            lbl_meta_dir=strip_trailing_slash(yaml_cfg["lfss"]["meta_en_dir"])
        
        with open(args.outfile, "a", encoding="utf-8") as f_out, \
            open(args.translatefile, "a", encoding="utf-8") as f_translate, \
            open(args.failfile, "a", encoding="utf-8") as f_fail:
            
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(task, idx, model, lbl_meta_dir, args): idx for idx in pending}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="task", dynamic_ncols=True):
                    out = future.result()
                    if out:
                        case_en = out["case_en"]
                        result = out["res"]
                        if case_en:
                            if "failed" in case_en or (isinstance(case_en, dict)and "failed" in case_en[list(case_en.keys())[0]]):
                                f_fail.write(json.dumps(case_en, ensure_ascii=False) + "\n")
                                f_fail.flush()
                            else:
                                f_translate.write(json.dumps(case_en, ensure_ascii=False) + "\n")
                                f_translate.flush()
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