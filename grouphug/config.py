from transformers.utils.logging import get_logger, set_verbosity

logger = get_logger("transformers")

# conventions on feature names
TEXT_VAR = "text"  # text columns are called <prefix>TEXT_VAR
INPUT_IDS_VAR = "input_ids"  # token ids are called <prefix>INPUT_IDS_VAR
INPUT_EMBEDDING_VAR = "embedding"  # embedding vars are called <prefix>INPUT_EMBEDDING_VAR
MASKED_PREFIX = "masked_"
MASKED_INPUT_IDS_VAR = f"{MASKED_PREFIX}{INPUT_IDS_VAR}"
MLM_LABELS_VAR = "mlm_labels"
MTD_LABELS_VAR = "mtd_labels"
CLM_LABELS_VAR = "clm_labels"

MTD_TOKEN_SIMILARITY = "token_similarity"
MTD_TOKEN_RANDOM = "random"

# essentially what _pad cares about
TOKENIZER_VARS = [INPUT_IDS_VAR, "attention_mask", "token_type_ids", "special_tokens_mask"]

# for labels and losses
IGNORE_INDEX = -100

# for saving and loading models
HEADS_FILE_NAME = "head_configs.json"
FORMATTER_FILE_NAME = "formatter.json"

# for random splits
DEFAULT_SEED = 42
