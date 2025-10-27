# QuickStart: Benchmark Data Generation

This guide shows how to generate benchmark data for **VQA**, **Classification**, and **Image Captioning** tasks using the MetaDent Benchmark framework.

> **Note** : Only **LLM-based models** are supported for data generation (e.g., `openai/gpt-oss-120b`).

You first need to configure the `config.yaml` file, for example:
```bash
api_models:
  openai/gpt-oss-120b:
    model_dir: /workspace/share/openai/gpt-oss-120b
    served_model_name: gpt-oss-120b
```

The data were generated using `openai/gpt-oss-120b`. To reproduce this setup, first download the [open-source model weights (gpt-oss-120b)](https://huggingface.co/openai/gpt-oss-120b) to your local directory (e.g., `/workspace/share/openai/gpt-oss-120b`), and then deploy the model locally via **vLLM**. Example command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /workspace/share/openai/gpt-oss-120b 
    --port 8000 \
    --served-model-name gpt-oss-120b \
    --gpu-memory-utilization 0.9 \
    --max_model_len 81920 \
    --tensor_parallel_size 4
```

## VQA Data Generation
```bash
python -m src.main \
    --task generation \
    --subtask vqa \
    --client_type api \
    --model_name openai/gpt-oss-120b \
    --api_base_url "http://localhost:8000/v1" \
    --api_key "EMPTY"
```

## Classification Data Generation
```bash
python -m src.main \
    --task generation \
    --subtask classification \
    --client_type api \
    --model_name openai/gpt-oss-120b \
    --api_base_url "http://localhost:8000/v1" \
    --api_key "EMPTY"
```

## Image Captioning Data Generation

```bash
python -m src.main \
    --task generation \
    --subtask captioning \
    --client_type api \
    --model_name openai/gpt-oss-120b \
    --api_base_url "http://localhost:8000/v1" \
    --api_key "EMPTY"
```

## Notes
- Both locally deployed models and API-based models are supported.
- Make sure the vLLM server is running and accessible at the specified api_base_url.
- **Make sure your model environment (local or API) is properly configured before running the scripts.**
- You can adjust `--start`, `--end`, and `--lfss_meta_type` parameters based on your dataset configuration.
- Generated benchmark files will be stored automatically under the `data/generation` directory.
- For configuration setup, see config.yaml and environment variable notes in the main README.