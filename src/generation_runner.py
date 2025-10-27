import os
import shutil
from collections import defaultdict

from tqdm import tqdm

from src.models.load_model import load_model
from src.utils.file_io import change_path_suffix, load_data, save_json_data


def run(args, yaml_cfg, model_cfg):
    model = load_model(args, model_cfg)

    if args.subtask == "vqa":
        from src.tasks.vqa.generator import run_vqa_generation
        run_vqa_generation(model, yaml_cfg, args)
        shutil.copy(change_path_suffix(args.outfile, ".json"), os.path.join(args.project_root, args.save_root_dir, "vqa.json"))
    
    elif args.subtask == "classification":
        from src.tasks.classification.generator import run_classification_generation
        run_classification_generation(model, yaml_cfg, args)
        
        label_json = load_data(change_path_suffix(args.outfile, ".json"))
        classification_json = defaultdict(list)
        labels = set(["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18"])
        
        for key, value in tqdm(label_json.items(), total=len(label_json), dynamic_ncols=True, desc=f"[{args.model_name}] Generating classification.json"):
            ids = set()
            for item in value:
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
                    s = str(value)
                    for l in labels:
                        if l in s:
                            ids.add(l)
                    break
            classification_json[key] = list(ids)
            classification_json[key].sort(key=lambda x: int(x[1:]))
        
        classification_json = {k: classification_json[k] for k in sorted(classification_json.keys())}
        save_json_data(classification_json, os.path.join(args.project_root, args.save_root_dir), "classification.json", title=f"{args.task} - {args.subtask} - {args.model_name}")
    
    elif args.subtask == "captioning":
        from src.tasks.captioning.generator import run_captioning_generation
        run_captioning_generation(model, yaml_cfg, args)
        shutil.copy(change_path_suffix(args.outfile, ".json"), os.path.join(args.project_root, args.save_root_dir, "captioning.json"))
