from src.models.load_model import load_model
from src.utils.common_utils import *


def run(args, yaml_cfg, model_cfg):
    model = load_model(args, model_cfg)

    if args.subtask == "vqa":
        from src.tasks.vqa.predictor import run_vqa_prediction
        run_vqa_prediction(model, yaml_cfg, args)
    
    elif args.subtask == "classification":
        from src.tasks.classification.predictor import run_classification_prediction
        run_classification_prediction(model, yaml_cfg, args)
    
    elif args.subtask == "captioning":
        from src.tasks.captioning.predictor import run_captioning_prediction
        run_captioning_prediction(model, yaml_cfg, args)
