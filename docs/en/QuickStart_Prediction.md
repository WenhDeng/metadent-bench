# QuickStart: VLM Prediction

This guide explains how to run **prediction tasks** using various Vision-Language Models (VLMs) within the MetaDent Benchmark framework.
The model generates answers based solely on the **input image** and **task-specific prompt**, covering three subtasks: **VQA**, **Classification**, and **Image Captioning**.

In our experiments, we locally deployed two models: [Baichuan-Omni-1d5](https://huggingface.co/baichuan-inc/Baichuan-Omni-1d5) and [Ovis2-34B](https://www.modelscope.cn/models/AIDC-AI/Ovis2-34B).
You need to download the weights for both models and specify their paths in the `config.yaml` file.

Since **vLLM** does not yet support Baichuan-Omni-1d5, our code provides a built-in deployment interface. You should add the following configuration under `local_models`:

```bash
local_models:
  baichuan-inc/Baichuan-Omni-1d5:
    model_dir: /workspace/share/baichuan-inc/Baichuan-Omni-1d5
```

You also need to set up a new conda environment according to the requirements of **Baichuan-Omni-1d5** and install its dependencies. For details, please refer to the [Baichuan-Omni-1.5 environment setup guide](https://github.com/baichuan-inc/Baichuan-Omni-1.5/).

For the **Ovis2-34B** model, we deployed it using **vLLM 0.9.1**. You should add the following configuration under `api_models`:

```bash
api_models:
  AIDC-AI/Ovis2-34B:
    model_dir: /workspace/share/AIDC-AI/Ovis2-34B
    served_model_name: ovis2-34B
```

and then deploy the model locally via **vLLM 0.9.1**. Example command:
```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve /workspace/share/AIDC-AI/Ovis2-34B \
    --port 8000 \
    --served-model-name Ovis2-34B \
    --gpu-memory-utilization 0.9 \
    --max_model_len 32768 \
    --tensor_parallel_size 2
```

For other models, we connect to them through third-party API services. Therefore, you only need to provide the `api_base_url` and `api_key`.


## VQA Prediction
1. **Baichuan-Omni-1d5**
    ```bash
    python -m src.main \
        --task prediction \
        --subtask vqa \
        --client_type local \
        --model_name baichuan-inc/Baichuan-Omni-1d5 \
        --gpus 0
    ```

2. **Ovis2-34B**
    ```bash
    python -m src.main \
        --task prediction \
        --subtask vqa \
        --client_type api \
        --model_name AIDC-AI/Ovis2-34B \
        --api_base_url "http://localhost:8000/v1" \
        --api_key "EMPTY"
    ```

3. **Other API-based models**
    ```bash
    python -m src.main \
        --task prediction \
        --subtask vqa \
        --client_type api \
        --model_name <model_name> \
        --api_base_url <api_base_url> \
        --api_key <api_key>
    ```

## Classification Prediction
The classification prediction process is similar to the VQA prediction process. You only need to change the `subtask` parameter to `classification`.

## Image Captioning Prediction
The image captioning prediction process is similar to the VQA prediction process. You only need to change the `subtask` parameter to `captioning`.

## Notes
- Both locally deployed models and API-based models are supported.
- Make sure the vLLM server is running and accessible at the specified api_base_url.
- **Make sure your model environment (local or API) is properly configured before running the scripts.**
- You can adjust `--start`, `--end`, and `--lfss_meta_type` parameters based on your dataset configuration.
- Generated files will be stored automatically under the `data/prediction/` directory.
- For configuration setup, see config.yaml and environment variable notes in the main README.