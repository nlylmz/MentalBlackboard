# Example to run Prediction Task
```
python -m open_source_evals.eval_prediction \
    --model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --task Prediction \
    --json-path <path to data> \ 
    --img-text-root <path to images>
```

# Example to run Prediction 2D Task
```
python -m open_source_evals.eval_prediction_2D \
    --model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --task Prediction_2D_images \
    --json-path <path to data> \ 
    --img-text-root <path to images>
```
# Example to run Prediction Video Task
```
python -m open_source_evals.eval_prediction_video \
    --model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --json-path <path to data> \ 
    --img-text-root <path to images>
```

# Example to run Planning Task
```
python -m open_source_evals.eval_planning_2D \
    --model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --task Planning_2D_images \
    --json-path <path to data> \ 
    --img-text-root <path to images>
```

# Example to run Generalization Task
```
python -m open_source_evals.eval_generalization_2D \
    --model-name llava-hf/llava-onevision-qwen2-7b-si-hf \
    --task Generalization_2D_images \
    --json-path <path to data> \ 
    --img-text-root <path to images>
```
