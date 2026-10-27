"""Qwen byte-level BPE tokenizer without Transformers or Torch imports."""
import json
from pathlib import Path
from tokenizers import Tokenizer, AddedToken, Regex, models, pre_tokenizers

SYSTEM_PREFIX = '<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n'
PROMPT_TEMPLATE = SYSTEM_PREFIX+'<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n'
# Qwen2Tokenizer's Unicode pre-tokenization expression.
PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

class QwenTokenizer:
    def __init__(self, directory):
        directory = Path(directory)
        self.tokenizer = Tokenizer(models.BPE.from_file(str(directory/'vocab.json'),str(directory/'merges.txt')))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(PATTERN),behavior='isolated'),
            pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False),
        ])
        config = json.loads((directory/'tokenizer_config.json').read_text())
        for key,value in sorted(config['added_tokens_decoder'].items(),key=lambda item:int(item[0])):
            token = AddedToken(value['content'],single_word=value.get('single_word',False),
                               lstrip=value.get('lstrip',False),rstrip=value.get('rstrip',False),
                               normalized=value.get('normalized',False),special=value.get('special',True))
            self.tokenizer.add_tokens([token])
            if self.tokenizer.token_to_id(value['content']) != int(key):
                raise ValueError(f'Token ID mismatch for {value["content"]}')

    def encode(self, text):
        return self.tokenizer.encode(text,add_special_tokens=False).ids
