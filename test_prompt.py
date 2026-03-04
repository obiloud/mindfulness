from transformers import AutoTokenizer

START_OF_HEADER = 128006
END_OF_HEADER = 128007

# Special control tokens
SOH_ID = 128259  # Start of Human turn
EOH_ID = 128260  # End of Human turn
SOA_ID = 128261  # Start of AI turn
EOA_ID = 128262  # End of AI turn (not used in maya1)
PAD_ID = 128263  # Padding token

# Text tokens
BOS_ID = 128000  # Begin of sequence (Llama BOS)
TEXT_EOT_ID = 128009  # End of text (appears in prefix, not a stop token!)

# Audio tokens
CODE_START_TOKEN_ID = 128257  # SOS - Start of Speech
CODE_END_TOKEN_ID = 128258   # EOS - End of Speech (audio stop token)
CODE_TOKEN_OFFSET = 128266   # Start of SNAC codes


tokenizer = AutoTokenizer.from_pretrained(
    "maya-research/maya1",
    trust_remote_code=True
)


description = "foo"
text = "bar"


prompt_ids = [
    BOS_ID,
    START_OF_HEADER,
    *tokenizer.encode("system", add_special_tokens=False),
    END_OF_HEADER,
    *tokenizer.encode(description, add_special_tokens=False),
    TEXT_EOT_ID,
    SOH_ID,
    *tokenizer.encode(text, add_special_tokens=False),
    TEXT_EOT_ID,
    EOH_ID,
    SOA_ID,
    CODE_START_TOKEN_ID
]
# Return as a list of IDs directly for vLLM

print(prompt_ids)