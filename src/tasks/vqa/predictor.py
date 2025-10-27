import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.models.base_model import BaseModel
from src.utils import prompt
from src.utils.common_utils import confirm_restart_if_exists, strip_trailing_slash
from src.utils.file_io import (
    convert_jsonl_to_json,
    load_completed_indices,
    load_vqa_data,
)


def task(idx: str, items: list, image_dir: str, model: BaseModel) -> dict:
    new_question_list = []
    
    for item in items:
        question_type = item["question_type"]
        question = item["question"]
        choice = item["choice"]
        
        image_path= os.path.join(image_dir, f"{idx}.png")
        p = prompt.vqa_answer_intraoral_condition.substitute(
            question=question,
            choice=model.j2t(choice),
            answer_options='"A", "B", "C", or "D"' if question_type == 'multiple_choice' else '"A" or "B"'
            )

        try:
            res = model.generate_from_image_and_text(image_path=image_path, prompt=p)
            AI_answer = res["answer"]
            AI_reason = res["reason"]
            new_question_list.append({
                "question_type": question_type,
                "question": question,
                "choice": choice,
                "answer": item["answer"],
                "reason": item["reason"],
                "AI_answer": AI_answer,
                "AI_reason": AI_reason
            })
        except Exception as e:
            new_question_list.append({
                "question_type": question_type,
                "question": question,
                "choice": choice,
                "answer": item["answer"],
                "reason": item["reason"],
                "AI_answer": None,
                "AI_reason": str(e)
            })
    
    return {
        "res": {idx: new_question_list}
    }

def run_vqa_prediction(model: BaseModel, yaml_cfg, args):
    # image_dir=yaml_cfg["lfss"]["image_dir"]
    image_dir=strip_trailing_slash(yaml_cfg["data"]["image_dir"])
    data = load_vqa_data()
    data_keys = set(data.keys())
    
    completed = load_completed_indices(args)
    completed = confirm_restart_if_exists(args, completed)
    all_indices = [f"{i:09d}" for i in range(args.start, args.end + 1)]
    pending = [idx for idx in all_indices if idx not in completed and idx in data_keys]
    tqdm.write(f"Total tasks: {len(all_indices)} | Completed: {len(completed)} | Skipped: {len(all_indices) - len(completed) - len(pending)} | Pending: {len(pending)}")
    
    if not pending:
        print("All tasks already completed.")
    else:
        with open(args.outfile, "a", encoding="utf-8") as f_out, \
            open(args.failfile, "a", encoding="utf-8") as f_fail:
            
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(task, idx, data[idx], image_dir, model): idx for idx in pending}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="task", dynamic_ncols=True):
                    out = future.result()
                    if out:
                        result = out["res"]
                        if result:
                            if "failed" in result or (isinstance(result, dict) and "failed" in result[list(result.keys())[0]]):
                                f_fail.write(json.dumps(result, ensure_ascii=False) + "\n")
                                f_fail.flush()
                            else:
                                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                                f_out.flush()
    
    convert_jsonl_to_json(args, jsonl_type="results")
    convert_jsonl_to_json(args, jsonl_type="failures")