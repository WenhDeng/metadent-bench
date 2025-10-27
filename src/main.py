import os

from src import evaluation_runner, generation_runner, prediction_runner
from src.utils.config_loader import load_args, load_model_config, load_yaml_config

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if __name__ == '__main__':
    args = load_args()
    if args.start > args.end:
        raise ValueError(f"Invalid range: start ({args.start}) must be <= end ({args.end})")
    
    if args.client_type == "api":
        if not args.api_base_url:
            args.api_base_url = os.getenv("API_BASE_URL")
            if not args.api_base_url:
                raise ValueError("--api_base_url is required for API mode")
        if not args.api_key:
            args.api_key = os.getenv("API_KEY")
            if not args.api_key:
                raise ValueError("--api_key is required for API mode")
    else:
        args.workers = 1
    
    if args.task == "evaluation" and args.subtask == "captioning" and not args.evaluator_model_name:
        raise ValueError("--evaluator_model_name is required for evaluation task and subtask is captioning.")
    
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    args.project_root = project_root
    args.save_root_dir = "data"
    
    if args.task == "generation" or args.task == "prediction" or (args.task == "evaluation" and args.subtask == "captioning"):
        args.outfile = os.path.join(args.save_root_dir, args.task, args.subtask, args.model_name, "results.jsonl")
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
        args.translatefile = os.path.join(args.save_root_dir, args.task, args.subtask, args.model_name, "translate.jsonl")
        os.makedirs(os.path.dirname(args.translatefile), exist_ok=True)
        args.failfile = os.path.join(args.save_root_dir, args.task, args.subtask, args.model_name, "failures.jsonl")
        os.makedirs(os.path.dirname(args.failfile), exist_ok=True)
    
        if os.path.exists(args.failfile):
            os.remove(args.failfile)
            print(f"Deleted '{args.failfile}'.")
        
        # captioning
        args.refinefile = os.path.join(args.save_root_dir, args.task, args.subtask, args.model_name, "refine.jsonl")
        os.makedirs(os.path.dirname(args.refinefile), exist_ok=True)
    
    yaml_cfg = load_yaml_config(args=args)
    model_cfg = load_model_config(args)
    args.served_model_name = model_cfg.get("served_model_name") if (model_cfg and model_cfg.get("served_model_name")) else args.model_name
    
    if args.task == "generation":
        generation_runner.run(args, yaml_cfg, model_cfg)
    elif args.task == "prediction":
        prediction_runner.run(args, yaml_cfg, model_cfg)
    elif args.task == "evaluation":
        evaluation_runner.run(args, yaml_cfg, model_cfg)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    print("Done!")
    