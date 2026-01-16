
### Model Files
If you performed **full fine-tuning (or just downloaded a base HF model)**, 




| File                    | Purpose                                               |
| ----------------------- | ----------------------------------------------------- |
| `config.json`           | Model architecture (layers, hidden size, heads, etc.) |
| `tokenizer.json`        | Tokenizer logic (fast tokenizer graph)                |
| `tokenizer_config.json` | Tokenizer settings (padding, truncation)              |
| `vocab.json`            | Token → ID mapping                                    |
| `added_tokens.json`     | Custom / special tokens                               |
| pytorch_model-00001-of-0000N.bin          |      **All model weights are present (sharded bins)**                        |
| pytorch_model-00002-of-0000N.bin    |                             |

 
  Model can be loaded and used for inference directly.

 
 

If  PEFT / LoRA / QLoRA  was used, then  

You would ALSO need:

```
adapter_config.json
adapter_model.safetensors (or .bin)
```

during inference:

* Base model weights are loaded first
* Adapter weights are applied on top



`*.bin` shards are OK

Hugging Face automatically shards large models:

```
pytorch_model-00001-of-00008.bin
```

This is **normal and expected** for large LLMs.
 

 

Sometimes you may ALSO see:

* `special_tokens_map.json`
* `generation_config.json`

These are **optional** and do not block inference.
  
More :
* What SageMaker uploads to `model.tar.gz`
* Why adapters are preferred in production
* How to validate model completeness programmatically

