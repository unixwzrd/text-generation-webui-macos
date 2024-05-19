import json
import re
from pathlib import Path

import yaml

from modules import chat, loaders, metadata_gguf, shared, ui


def get_fallback_settings():
    return {
        'wbits': 'None',
        'groupsize': 'None',
        'desc_act': False,
        'model_type': 'None',
        'max_seq_len': 2048,
        'n_ctx': 2048,
        'rope_freq_base': 0,
        'compress_pos_emb': 1,
        'truncation_length': shared.settings['truncation_length'],
        'skip_special_tokens': shared.settings['skip_special_tokens'],
        'custom_stopping_strings': shared.settings['custom_stopping_strings'],
    }


def get_model_metadata(model):
    model_settings = {}

    # Get settings from models/config.yaml and models/config-user.yaml
    settings = shared.model_config
    for pat in settings:
        if re.match(pat.lower(), model.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    path = Path(f'{shared.args.model_dir}/{model}/config.json')
    if path.exists():
        hf_metadata = json.loads(open(path, 'r', encoding='utf-8').read())
    else:
        hf_metadata = None

    if 'loader' not in model_settings:
        if hf_metadata is not None and 'quip_params' in hf_metadata:
            loader = 'QuIP#'
        else:
            loader = infer_loader(model, model_settings)

        model_settings['loader'] = loader

    # GGUF metadata
    if model_settings['loader'] in ['llama.cpp', 'llamacpp_HF']:
        path = Path(f'{shared.args.model_dir}/{model}')
        if path.is_file():
            model_file = path
        else:
            model_file = list(path.glob('*.gguf'))[0]

        metadata = metadata_gguf.load_metadata(model_file)

        for k in metadata:
            if k.endswith('context_length'):
                model_settings['n_ctx'] = metadata[k]
            elif k.endswith('rope.freq_base'):
                model_settings['rope_freq_base'] = metadata[k]
            elif k.endswith('rope.scale_linear'):
                model_settings['compress_pos_emb'] = metadata[k]
            elif k.endswith('block_count'):
                model_settings['n_gpu_layers'] = metadata[k] + 1

        if 'tokenizer.chat_template' in metadata:
            template = metadata['tokenizer.chat_template']
            eos_token = metadata['tokenizer.ggml.tokens'][metadata['tokenizer.ggml.eos_token_id']]
            bos_token = metadata['tokenizer.ggml.tokens'][metadata['tokenizer.ggml.bos_token_id']]
            template = template.replace('eos_token', "'{}'".format(eos_token))
            template = template.replace('bos_token', "'{}'".format(bos_token))

            template = re.sub(r'raise_exception\([^)]*\)', "''", template)
            template = re.sub(r'{% if add_generation_prompt %}.*', '', template, flags=re.DOTALL)
            model_settings['instruction_template'] = 'Custom (obtained from model metadata)'
            model_settings['instruction_template_str'] = template

    else:
        # Transformers metadata
        if hf_metadata is not None:
            metadata = json.loads(open(path, 'r', encoding='utf-8').read())
            for k in ['max_position_embeddings', 'model_max_length', 'max_seq_len']:
                if k in metadata:
                    model_settings['truncation_length'] = metadata[k]
                    model_settings['max_seq_len'] = metadata[k]

            if 'rope_theta' in metadata:
                model_settings['rope_freq_base'] = metadata['rope_theta']
            elif 'attn_config' in metadata and 'rope_theta' in metadata['attn_config']:
                model_settings['rope_freq_base'] = metadata['attn_config']['rope_theta']

            if 'rope_scaling' in metadata and type(metadata['rope_scaling']) is dict and all(key in metadata['rope_scaling'] for key in ('type', 'factor')):
                if metadata['rope_scaling']['type'] == 'linear':
                    model_settings['compress_pos_emb'] = metadata['rope_scaling']['factor']

            # Read GPTQ metadata for old GPTQ loaders
            if 'quantization_config' in metadata and metadata['quantization_config'].get('quant_method', '') != 'exl2':
                if 'bits' in metadata['quantization_config']:
                    model_settings['wbits'] = metadata['quantization_config']['bits']
                if 'group_size' in metadata['quantization_config']:
                    model_settings['groupsize'] = metadata['quantization_config']['group_size']
                if 'desc_act' in metadata['quantization_config']:
                    model_settings['desc_act'] = metadata['quantization_config']['desc_act']

        # Read AutoGPTQ metadata
        path = Path(f'{shared.args.model_dir}/{model}/quantize_config.json')
        if path.exists():
            metadata = json.loads(open(path, 'r', encoding='utf-8').read())
            if 'bits' in metadata:
                model_settings['wbits'] = metadata['bits']
            if 'group_size' in metadata:
                model_settings['groupsize'] = metadata['group_size']
            if 'desc_act' in metadata:
                model_settings['desc_act'] = metadata['desc_act']

    # Try to find the Jinja instruct template
    path = Path(f'{shared.args.model_dir}/{model}') / 'tokenizer_config.json'
    if path.exists():
        metadata = json.loads(open(path, 'r', encoding='utf-8').read())
        if 'chat_template' in metadata:
            template = metadata['chat_template']
            if isinstance(template, list):
                template = template[0]['template']

            for k in ['eos_token', 'bos_token']:
                if k in metadata:
                    value = metadata[k]
                    if type(value) is dict:
                        value = value['content']

                    template = template.replace(k, "'{}'".format(value))

            template = re.sub(r'raise_exception\([^)]*\)', "''", template)
            template = re.sub(r'{% if add_generation_prompt %}.*', '', template, flags=re.DOTALL)
            model_settings['instruction_template'] = 'Custom (obtained from model metadata)'
            model_settings['instruction_template_str'] = template

    if 'instruction_template' not in model_settings:
        model_settings['instruction_template'] = 'Alpaca'

    # Ignore rope_freq_base if set to the default value
    if 'rope_freq_base' in model_settings and model_settings['rope_freq_base'] == 10000:
        model_settings.pop('rope_freq_base')

    # Apply user settings from models/config-user.yaml
    settings = shared.user_config
    for pat in settings:
        if re.match(pat.lower(), model.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    # Load instruction template if defined by name rather than by value
    if model_settings['instruction_template'] != 'Custom (obtained from model metadata)':
        model_settings['instruction_template_str'] = chat.load_instruction_template(model_settings['instruction_template'])

    return model_settings


def infer_loader(model_name, model_settings):
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    if not path_to_model.exists():
        loader = None
    elif (path_to_model / 'quantize_config.json').exists() or ('wbits' in model_settings and type(model_settings['wbits']) is int and model_settings['wbits'] > 0):
        loader = 'ExLlamav2_HF'
    elif (path_to_model / 'quant_config.json').exists() or re.match(r'.*-awq', model_name.lower()):
        loader = 'AutoAWQ'
    elif len(list(path_to_model.glob('*.gguf'))) > 0 and path_to_model.is_dir() and (path_to_model / 'tokenizer_config.json').exists():
        loader = 'llamacpp_HF'
    elif len(list(path_to_model.glob('*.gguf'))) > 0:
        loader = 'llama.cpp'
    elif re.match(r'.*\.gguf', model_name.lower()):
        loader = 'llama.cpp'
    elif re.match(r'.*exl2', model_name.lower()):
        loader = 'ExLlamav2_HF'
    elif re.match(r'.*-hqq', model_name.lower()):
        return 'HQQ'
    else:
        loader = 'Transformers'

    return loader


def update_model_parameters(state, initial=False):
    '''
    UI: update the command-line arguments based on the interface values
    '''
    elements = ui.list_model_elements()  # the names of the parameters
    gpu_memories = []

    for i, element in enumerate(elements):
        if element not in state:
            continue

        value = state[element]
        if element.startswith('gpu_memory'):
            gpu_memories.append(value)
            continue

        if initial and element in shared.provided_arguments:
            continue

        # Setting null defaults
        if element in ['wbits', 'groupsize', 'model_type'] and value == 'None':
            value = vars(shared.args_defaults)[element]
        elif element in ['cpu_memory'] and value == 0:
            value = vars(shared.args_defaults)[element]

        # Making some simple conversions
        if element in ['wbits', 'groupsize', 'pre_layer']:
            value = int(value)
        elif element == 'cpu_memory' and value is not None:
            value = f"{value}MiB"

        if element in ['pre_layer']:
            value = [value] if value > 0 else None

        setattr(shared.args, element, value)

    found_positive = False
    for i in gpu_memories:
        if i > 0:
            found_positive = True
            break

    if not (initial and vars(shared.args)['gpu_memory'] != vars(shared.args_defaults)['gpu_memory']):
        if found_positive:
            shared.args.gpu_memory = [f"{i}MiB" for i in gpu_memories]
        else:
            shared.args.gpu_memory = None


def apply_model_settings_to_state(model, state):
    '''
    UI: update the state variable with the model settings
    '''
    model_settings = get_model_metadata(model)
    if 'loader' in model_settings:
        loader = model_settings.pop('loader')

        # If the user is using an alternative loader for the same model type, let them keep using it
        if not (loader == 'ExLlamav2_HF' and state['loader'] in ['GPTQ-for-LLaMa', 'ExLlamav2', 'AutoGPTQ']):
            state['loader'] = loader

    for k in model_settings:
        if k in state:
            if k in ['wbits', 'groupsize']:
                state[k] = str(model_settings[k])
            else:
                state[k] = model_settings[k]

    return state


def save_model_settings(model, state):
    '''
    Save the settings for this model to models/config-user.yaml
    '''
    if model == 'None':
        yield ("Not saving the settings because no model is selected in the menu.")
        return

    user_config = shared.load_user_config()
    model_regex = model + '$'  # For exact matches
    if model_regex not in user_config:
        user_config[model_regex] = {}

    for k in ui.list_model_elements():
        if k == 'loader' or k in loaders.loaders_and_params[state['loader']]:
            user_config[model_regex][k] = state[k]

    shared.user_config = user_config

    output = yaml.dump(user_config, sort_keys=False)
    p = Path(f'{shared.args.model_dir}/config-user.yaml')
    with open(p, 'w') as f:
        f.write(output)

    yield (f"Settings for `{model}` saved to `{p}`.")


def save_instruction_template(model, template):
    '''
    Similar to the function above, but it saves only the instruction template.
    '''
    if model == 'None':
        yield ("Not saving the template because no model is selected in the menu.")
        return

    user_config = shared.load_user_config()
    model_regex = model + '$'  # For exact matches
    if model_regex not in user_config:
        user_config[model_regex] = {}

    if template == 'None':
        user_config[model_regex].pop('instruction_template', None)
    else:
        user_config[model_regex]['instruction_template'] = template

    shared.user_config = user_config

    output = yaml.dump(user_config, sort_keys=False)
    p = Path(f'{shared.args.model_dir}/config-user.yaml')
    with open(p, 'w') as f:
        f.write(output)

    if template == 'None':
        yield (f"Instruction template for `{model}` unset in `{p}`, as the value for template was `{template}`.")
    else:
        yield (f"Instruction template for `{model}` saved to `{p}` as `{template}`.")
