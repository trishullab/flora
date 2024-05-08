# flora

`peft` contains the code of flora under `peft/src/peft/tuners/flora.py`

To start,
```python
from transformers import AutoModel
from peft import get_peft_config, get_peft_model, FLoraConfig, TaskType
model_name_or_path = YOUR_MODEL
tokenizer_name_or_path = YOUR_TOKENIZER

peft_config = FLoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=2, lora_alpha=16
)

model = AutoModel.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

This is based on the `peft` version 0.4.0. We are working on the pull request to merge it to the latest `peft`.
