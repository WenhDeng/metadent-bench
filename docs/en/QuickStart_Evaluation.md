# QuickStart: Evaluation

This section provides example commands for running the **evaluation** task on different subtasks — **VQA**, **Classification**, and **Image Captioning**.
Please make sure you have properly configured the environment and `config.yaml` file before execution.

## VQA
1.  **Baichuan-Omni-1d5**
    ```bash
    python -m src.main \
        --task evaluation \
        --subtask vqa \
        --model_name baichuan-inc/Baichuan-Omni-1d5
    ```

2.  **Ovis2-34B**
    ```bash
    python -m src.main \
        --task evaluation \
        --subtask vqa \
        --model_name AIDC-AI/Ovis2-34B
    ```

3.  **Other models**
    ```bash
    python -m src.main \
        --task evaluation \
        --subtask vqa \
        --model_name <model_name>
    ```

## Classification
The classification evaluation process is similar to the VQA evaluation process. You only need to change the `subtask` parameter to `classification`.

## Image Captioning
> **Note 1** :
> Image captioning evaluation requires an additional **LLM-based evaluator** to assess the generated captions.
> You must specify the evaluator model using the `--evaluator_model_name` argument.
In our setup, we use `openai/gpt-oss-120b` as the evaluator.

> **Note 2** :
> Additionally, the **BERTScore** evaluation depends on **RoBERTa-large**, which typically requires ~4GB of GPU memory.
> It is recommended to specify the GPU for evaluation via the `--gpus` parameter.

Before running captioning evaluations, you must register the evaluator model in your `config.yaml`. For example:
```bash
api_models:
  openai/gpt-oss-120b:
    model_dir: /workspace/share/openai/gpt-oss-120b
    served_model_name: gpt-oss-120b
```

Download the [open-source model weights (gpt-oss-120b)](https://huggingface.co/openai/gpt-oss-120b) and deploy them locally using **vLLM**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /workspace/share/openai/gpt-oss-120b 
    --port 8000 \
    --served-model-name gpt-oss-120b \
    --gpu-memory-utilization 0.9 \
    --max_model_len 81920 \
    --tensor_parallel_size 4
```

1.  **Baichuan-Omni-1d5**
    ```bash
    python -m src.main \
        --task evaluation \
        --subtask captioning \
        --client_type api \
        --model_name baichuan-inc/Baichuan-Omni-1d5 \
        --evaluator_model_name openai/gpt-oss-120b \
        --api_base_url "http://localhost:8000/v1" \
        --api_key "EMPTY" \
        --gpus 0
    ```

2.  **Ovis2-34B**
    ```bash
    python -m src.main \
        --task evaluation \
        --subtask captioning \
        --client_type api \
        --model_name AIDC-AI/Ovis2-34B \
        --evaluator_model_name openai/gpt-oss-120b \
        --api_base_url "http://localhost:8000/v1" \
        --api_key "EMPTY" \
        --gpus 0
    ```

3.  **Other models**
    ```bash
    python -m src.main \
        --task evaluation \
        --subtask captioning \
        --client_type api \
        --model_name <model_name> \
        --evaluator_model_name openai/gpt-oss-120b \
        --api_base_url "http://localhost:8000/v1" \
        --api_key "EMPTY" \
        --gpus 0
    ```


## Notes
- Both locally deployed models and API-based models are supported.
- Make sure the vLLM server is running and accessible at the specified api_base_url.
- **Make sure your model environment (local or API) is properly configured before running the scripts.**
- You can adjust `--start`, `--end`, and `--lfss_meta_type` parameters based on your dataset configuration.
- Generated benchmark files will be stored automatically under the `data/prediction` directory.
- For configuration setup, see config.yaml and environment variable notes in the main README.