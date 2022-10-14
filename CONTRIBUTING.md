# Feature requests and PRs

If you want to add a complex feature or change which is not already mentioned, please open an issue or discussion topic to discuss. 
For simple additions and bug fixes you can open a PR directly.

## Formatting and pre-commit hooks

To ensure your PR is properly formatted, install pre-commit hooks using `pre-commit install`

This will run black, isort, and clear any output from example notebooks when committing.

# Notes on Grouphug internals

This section contains notes on implementation details of huggingface transformers and grouphug.

## Computing metrics

Computing metrics has been changed to be passed extra parameters, allowing the metrics function to know what data is passed.
The function in examples/utils works as a fairly generic implementation and could be added as a default in future versions.

# Notes on Huggingface Transformers internals

These are largely my own notes on the internals of the transformers package and how they interact.

## Tokenizers

* Tokenizers have `.model_input_names` to determined what to pad, e.g. `['input_ids, 'token_type_ids','attention_mask']`
  * However, these are mostly ignored except for the first, and `_pad` has a hardcoded check for `['input_ids, 'token_type_ids','attention_mask','special_tokens_mask']`
* Tokenizers have model_max_len, which is often unset and left at it's default of LARGE_INTEGER
* Dynamic padding is done by various collators via Tokenizer.pad, but this does not truncate.

## Trainer

* Model outputs are ordered dicts
* All keys not named 'loss' are assumed to be logits
  * Somehow one of the GPT models still returns two losses. 
