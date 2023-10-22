import random
from pathlib import Path

import torch
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer
)
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler

from modules import shared
from modules.logging_colors import logger
from modules.text_generation import get_max_prompt_length

try:
    import flash_attn
except ModuleNotFoundError:
    logger.warning(
        'You are running ExLlamaV2 without flash-attention. This will cause the VRAM usage '
        'to be a lot higher than it could be.\n'
        'Try installing flash-attention following the instructions here: '
        'https://github.com/Dao-AILab/flash-attention#installation-and-features'
    )
    pass


class Exllamav2Model:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path_to_model):

        path_to_model = Path(f'{shared.args.model_dir}') / Path(path_to_model)

        config = ExLlamaV2Config()
        config.model_dir = str(path_to_model)
        config.prepare()

        config.max_seq_len = shared.args.max_seq_len
        config.scale_pos_emb = shared.args.compress_pos_emb
        config.scale_alpha_value = shared.args.alpha_value

        model = ExLlamaV2(config)

        split = None
        if shared.args.gpu_split:
            split = [float(alloc) for alloc in shared.args.gpu_split.split(",")]

        model.load(split)

        tokenizer = ExLlamaV2Tokenizer(config)
        cache = ExLlamaV2Cache(model)
        generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

        result = self()
        result.model = model
        result.cache = cache
        result.tokenizer = tokenizer
        result.generator = generator
        return result, result

    def encode(self, string, **kwargs):
        return self.tokenizer.encode(string, add_bos=True)

    def decode(self, ids, **kwargs):
        if isinstance(ids, list):
            ids = torch.tensor([ids])
        elif isinstance(ids, torch.Tensor) and ids.numel() == 1:
            ids = ids.view(1, -1)

        return self.tokenizer.decode(ids)[0]

    def get_logits(self, token_ids, **kwargs):
        self.cache.current_seq_len = 0
        self.model.forward(token_ids[:, :-1], self.cache, input_mask=None, preprocess_only=True)
        return self.model.forward(token_ids[:, -1:], self.cache, input_mask=None, **kwargs).float().cpu()

    def generate_with_streaming(self, prompt, state):
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = state['temperature']
        settings.top_k = state['top_k']
        settings.top_p = state['top_p']
        settings.token_repetition_penalty = state['repetition_penalty']
        settings.token_repetition_range = -1 if state['repetition_penalty_range'] <= 0 else state['repetition_penalty_range']
        if state['ban_eos_token']:
            settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])

        if state['custom_token_bans']:
            to_ban = [int(x) for x in state['custom_token_bans'].split(',')]
            if len(to_ban) > 0:
                settings.disallow_tokens(self.tokenizer, to_ban)

        ids = self.tokenizer.encode(prompt, add_bos=state['add_bos_token'])
        ids = ids[:, -get_max_prompt_length(state):]
        initial_len = ids.shape[-1]

        if state['auto_max_new_tokens']:
            max_new_tokens = state['truncation_length'] - ids.shape[-1]
        else:
            max_new_tokens = state['max_new_tokens']

        # _gen_begin_base
        self.cache.current_seq_len = 0
        self.model.forward(ids[:, :-1], self.cache, input_mask=None, preprocess_only=True)

        has_leading_space = False
        for i in range(max_new_tokens):
            logits = self.model.forward(ids[:, -1:], self.cache, input_mask=None).float().cpu()
            token, _ = ExLlamaV2Sampler.sample(logits, settings, ids, random.random())
            ids = torch.cat([ids, token], dim=1)

            if i == 0 and self.tokenizer.tokenizer.IdToPiece(int(token)).startswith('▁'):
                has_leading_space = True

            decoded_text = self.tokenizer.decode(ids[:, initial_len:])[0]
            if has_leading_space:
                decoded_text = ' ' + decoded_text

            yield decoded_text

            if token.item() == self.tokenizer.eos_token_id or shared.stop_everything:
                break

    def generate(self, prompt, state):
        output = ''
        for output in self.generate_with_streaming(prompt, state):
            pass

        return output
