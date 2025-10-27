import argparse
import os

import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
config_dir = os.path.join(project_root, "configs")


def load_yaml_config(args: argparse.Namespace):
    if args.test_mode:
        yaml_path=os.path.join(config_dir, "config-dev.yaml")
    else:
        yaml_path=os.path.join(config_dir, "config.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_model_config(args: argparse.Namespace):
    cfg = load_yaml_config(args=args)
    if args.task == "evaluation" and args.subtask == "captioning":
        model_name = args.evaluator_model_name
    else:
        model_name = args.model_name
    
    if args.client_type == "local":
        model_cfg = cfg["local_models"].get(model_name)
    elif args.client_type == "api":
        model_cfg = cfg["api_models"].get(model_name)
    if args.client_type == "local" and not model_cfg:
        raise ValueError(f"Model '{model_name}' not found in config.yaml")
    return model_cfg

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path to use (for local or API mode).")
    parser.add_argument("--evaluator_model_name", type=str, default=None, help="Name or path of the LLM used for evaluation. This parameter is required only when the task is 'evaluation' and the subtask is 'captioning'. The evaluator model must be available either locally or via an API, and it will be used to assess the generated captions.")
    parser.add_argument("--do_sample", action="store_true", default=False, help="Whether to sample from the model.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling. Higher temperature results in more random output.")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="Maximum number of tokens to generate.")
    parser.add_argument("--client_type",  type=str, choices=["local", "api"], default="api", help="Choose how to load the model: 'local' for local weights, or 'api' for remote (OpenAI/vLLM-compatible) endpoints.")
    
    parser.add_argument("--gpus", type=str, default=None, help="GPUs to use (comma-separated), e.g. 0,1,2,3. If not specified, all GPUs will be used.")
    parser.add_argument("--api_base_url", type=str, default=None, help="Base URL for the OpenAI-compatible API (e.g. http://localhost:8000/v1)")
    parser.add_argument("--api_key", type=str, default=None, help="API key for the OpenAI-compatible API.")
    
    parser.add_argument("--task", type=str, required=True, choices=["generation", "prediction", "evaluation"], help="Main task type: 'generation' to generate data, 'prediction' to run VLM, 'evaluation' to compute evaluation metrics")
    parser.add_argument("--subtask", type=str, required=True, choices=["vqa", "classification", "captioning"], help="Sub-task type")
    
    parser.add_argument("--start", type=int, default=1, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=100, help="End index (inclusive)")
    parser.add_argument("--workers", type=int, default=8, help="Number of threads to use")
    
    parser.add_argument("--lfss_meta_type", type=str, choices=["cn", "en"], default="en", help="Language of the meta data, cn or en")
    parser.add_argument("--test_mode", action="store_true", default=False, help="Whether to run in test mode (use private data)")
    
    # captioning
    parser.add_argument("--chunk", action="store_true", help="Whether to chunk the data into smaller batches to avoid GPU memory issues")
    parser.add_argument("--chunk_size", type=int, default=512, help="Size of each chunk")
    return parser.parse_args()
