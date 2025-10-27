import os
from collections import defaultdict

from tqdm import tqdm

from src.models.load_model import load_model
from src.utils.common_utils import strip_trailing_slash
from src.utils.file_io import save_json_data
from src.utils.lfss_io import InfoReader, LabelReader, SkipReader

LOW_CONFIDENCE_COUNT = 0
HIGH_CONFIDENCE_COUNT = 0

def task(idx: str, lbl_meta_dir: str, distribution_dict: dict):
    global LOW_CONFIDENCE_COUNT, HIGH_CONFIDENCE_COUNT
    
    r = SkipReader(lbl_meta_dir=lbl_meta_dir)
    skip = r.get(idx)
    if skip is not None:
        return None

    r = LabelReader(lbl_meta_dir=lbl_meta_dir)
    label = r.get(idx)
    if label is None:
        return None
    
    label = label.compact_json()
    
    # statistics the number of low-confidence and high-confidence labels in the abnormal labeling
    for item in label["items"]:
        if item["low_confidence"]:
            LOW_CONFIDENCE_COUNT += 1
        else:
            HIGH_CONFIDENCE_COUNT += 1
    
    r = InfoReader(lbl_meta_dir=lbl_meta_dir)
    info = r.get(idx)
    if info is None:
        raise FileExistsError(f"{idx} not found")
    info = info.compact_json()
    
    ds_name = info["source"]
    distribution_dict[ds_name].append(idx)

def run(args, yaml_cfg, model_cfg):
    # generate the distribution json file
    distribution_dict = defaultdict(list)
    all_indices = [f"{i:09d}" for i in range(args.start, args.end + 1)]
    
    if args.lfss_meta_type == "cn":
        lbl_meta_dir=strip_trailing_slash(yaml_cfg["lfss"]["meta_cn_dir"])
    else:
        lbl_meta_dir=strip_trailing_slash(yaml_cfg["lfss"]["meta_en_dir"])
    
    for idx in tqdm(all_indices, desc="Generating distribution.json", dynamic_ncols=True):
        task(idx, lbl_meta_dir, distribution_dict)
    
    tqdm.write(f"low_confidence_cnt: {LOW_CONFIDENCE_COUNT}, high_confidence_cnt: {HIGH_CONFIDENCE_COUNT}")
    save_json_data(distribution_dict, os.path.join(args.project_root, args.save_root_dir), "distribution.json")
    
    if args.subtask == "vqa":
        from src.tasks.vqa.evaluator import run_vqa_evaluation
        run_vqa_evaluation(args)
    elif args.subtask == "classification":
        from src.tasks.classification.evaluator import run_classification_evaluation
        run_classification_evaluation(args)
    elif args.subtask == "captioning":
        from src.tasks.captioning.evaluator import run_captioning_evaluation
        model = load_model(args, model_cfg)
        run_captioning_evaluation(model, yaml_cfg, args)
