import base64
import copy
import functools
import json
import re
from datetime import datetime
from pathlib import Path

import gradio as gr
import yaml
from PIL import Image

import modules.shared as shared
from modules.extensions import apply_extensions
from modules.html_generator import chat_html_wrapper, make_thumbnail
from modules.logging_colors import logger
from modules.text_generation import (
    generate_reply,
    get_encoded_length,
    get_max_prompt_length
)
from modules.utils import (
    delete_file,
    get_available_characters,
    replace_all,
    save_file
)


def get_turn_substrings(state, instruct=False):
    if instruct:
        if 'turn_template' not in state or state['turn_template'] == '':
            template = '<|user|>\n<|user-message|>\n<|bot|>\n<|bot-message|>\n'
        else:
            template = state['turn_template'].replace(r'\n', '\n')
    else:
        template = '<|user|>: <|user-message|>\n<|bot|>: <|bot-message|>\n'

    replacements = {
        '<|user|>': state['name1_instruct' if instruct else 'name1'].strip(),
        '<|bot|>': state['name2_instruct' if instruct else 'name2'].strip(),
    }

    output = {
        'user_turn': template.split('<|bot|>')[0],
        'bot_turn': '<|bot|>' + template.split('<|bot|>')[1],
        'user_turn_stripped': template.split('<|bot|>')[0].split('<|user-message|>')[0],
        'bot_turn_stripped': '<|bot|>' + template.split('<|bot|>')[1].split('<|bot-message|>')[0],
    }

    for k in output:
        output[k] = replace_all(output[k], replacements)

    return output


def generate_chat_prompt(user_input, state, **kwargs):
    impersonate = kwargs.get('impersonate', False)
    _continue = kwargs.get('_continue', False)
    also_return_rows = kwargs.get('also_return_rows', False)
    history = kwargs.get('history', state['history'])['internal']
    is_instruct = state['mode'] == 'instruct'

    # Find the maximum prompt size
    max_length = get_max_prompt_length(state)
    all_substrings = {
        'chat': get_turn_substrings(state, instruct=False),
        'instruct': get_turn_substrings(state, instruct=True)
    }

    substrings = all_substrings['instruct' if is_instruct else 'chat']

    # Create the template for "chat-instruct" mode
    if state['mode'] == 'chat-instruct':
        wrapper = ''
        command = state['chat-instruct_command'].replace('<|character|>', state['name2'] if not impersonate else state['name1'])
        wrapper += state['context_instruct']
        wrapper += all_substrings['instruct']['user_turn'].replace('<|user-message|>', command)
        wrapper += all_substrings['instruct']['bot_turn_stripped']
        if impersonate:
            wrapper += substrings['user_turn_stripped'].rstrip(' ')
        elif _continue:
            wrapper += apply_extensions('bot_prefix', substrings['bot_turn_stripped'], state)
            wrapper += history[-1][1]
        else:
            wrapper += apply_extensions('bot_prefix', substrings['bot_turn_stripped'].rstrip(' '), state)
    else:
        wrapper = '<|prompt|>'

    # Build the prompt
    min_rows = 3
    i = len(history) - 1
    rows = [state['context_instruct'] if is_instruct else f"{state['context'].strip()}\n"]
    while i >= 0 and get_encoded_length(wrapper.replace('<|prompt|>', ''.join(rows))) < max_length:
        if _continue and i == len(history) - 1:
            if state['mode'] != 'chat-instruct':
                rows.insert(1, substrings['bot_turn_stripped'] + history[i][1].strip())
        else:
            rows.insert(1, substrings['bot_turn'].replace('<|bot-message|>', history[i][1].strip()))

        string = history[i][0]
        if string not in ['', '<|BEGIN-VISIBLE-CHAT|>']:
            rows.insert(1, replace_all(substrings['user_turn'], {'<|user-message|>': string.strip(), '<|round|>': str(i)}))

        i -= 1

    if impersonate:
        if state['mode'] == 'chat-instruct':
            min_rows = 1
        else:
            min_rows = 2
            rows.append(substrings['user_turn_stripped'].rstrip(' '))
    elif not _continue:
        # Add the user message
        if len(user_input) > 0:
            rows.append(replace_all(substrings['user_turn'], {'<|user-message|>': user_input.strip(), '<|round|>': str(len(history))}))

        # Add the character prefix
        if state['mode'] != 'chat-instruct':
            rows.append(apply_extensions('bot_prefix', substrings['bot_turn_stripped'].rstrip(' '), state))

    while len(rows) > min_rows and get_encoded_length(wrapper.replace('<|prompt|>', ''.join(rows))) >= max_length:
        rows.pop(1)

    prompt = wrapper.replace('<|prompt|>', ''.join(rows))
    if also_return_rows:
        return prompt, rows
    else:
        return prompt


def get_stopping_strings(state):
    stopping_strings = []
    if state['mode'] in ['instruct', 'chat-instruct']:
        stopping_strings += [
            state['turn_template'].split('<|user-message|>')[1].split('<|bot|>')[0] + '<|bot|>',
            state['turn_template'].split('<|bot-message|>')[1] + '<|user|>'
        ]

        replacements = {
            '<|user|>': state['name1_instruct'],
            '<|bot|>': state['name2_instruct']
        }

        for i in range(len(stopping_strings)):
            stopping_strings[i] = replace_all(stopping_strings[i], replacements).rstrip(' ').replace(r'\n', '\n')

    if state['mode'] in ['chat', 'chat-instruct']:
        stopping_strings += [
            f"\n{state['name1']}:",
            f"\n{state['name2']}:"
        ]

    if state['stop_at_newline']:
        stopping_strings.append("\n")

    return stopping_strings


def chatbot_wrapper(text, state, regenerate=False, _continue=False, loading_message=True):
    history = state['history']
    output = copy.deepcopy(history)
    output = apply_extensions('history', output)
    state = apply_extensions('state', state)
    if shared.model_name == 'None' or shared.model is None:
        logger.error("No model is loaded! Select one in the Model tab.")
        yield output
        return

    # Defining some variables
    just_started = True
    visible_text = None
    stopping_strings = get_stopping_strings(state)
    is_stream = state['stream']

    # Preparing the input
    if not any((regenerate, _continue)):
        visible_text = text
        text, visible_text = apply_extensions('chat_input', text, visible_text, state)
        text = apply_extensions('input', text, state)

        # *Is typing...*
        if loading_message:
            yield {'visible': output['visible'] + [[visible_text, shared.processing_message]], 'internal': output['internal']}
    else:
        text, visible_text = output['internal'][-1][0], output['visible'][-1][0]
        if regenerate:
            output['visible'].pop()
            output['internal'].pop()
            # *Is typing...*
            if loading_message:
                yield {'visible': output['visible'] + [[visible_text, shared.processing_message]], 'internal': output['internal']}
        elif _continue:
            last_reply = [output['internal'][-1][1], output['visible'][-1][1]]
            if loading_message:
                yield {'visible': output['visible'][:-1] + [[visible_text, last_reply[1] + '...']], 'internal': output['internal']}

    # Generating the prompt
    kwargs = {
        '_continue': _continue,
        'history': output,
    }

    prompt = apply_extensions('custom_generate_chat_prompt', text, state, **kwargs)
    if prompt is None:
        prompt = generate_chat_prompt(text, state, **kwargs)

    # Generate
    cumulative_reply = ''
    for i in range(state['chat_generation_attempts']):
        reply = None
        for j, reply in enumerate(generate_reply(prompt + cumulative_reply, state, stopping_strings=stopping_strings, is_chat=True)):
            reply = cumulative_reply + reply

            # Extract the reply
            visible_reply = re.sub("(<USER>|<user>|{{user}})", state['name1'], reply)

            # We need this global variable to handle the Stop event,
            # otherwise gradio gets confused
            if shared.stop_everything:
                output['visible'][-1][1] = apply_extensions('output', output['visible'][-1][1], state)
                yield output
                return

            if just_started:
                just_started = False
                if not _continue:
                    output['internal'].append(['', ''])
                    output['visible'].append(['', ''])

            if _continue:
                output['internal'][-1] = [text, last_reply[0] + reply]
                output['visible'][-1] = [visible_text, last_reply[1] + visible_reply]
                if is_stream:
                    yield output
            elif not (j == 0 and visible_reply.strip() == ''):
                output['internal'][-1] = [text, reply.lstrip(' ')]
                output['visible'][-1] = [visible_text, visible_reply.lstrip(' ')]
                if is_stream:
                    yield output

        if reply in [None, cumulative_reply]:
            break
        else:
            cumulative_reply = reply

    output['visible'][-1][1] = apply_extensions('output', output['visible'][-1][1], state)
    yield output


def impersonate_wrapper(text, start_with, state):
    if shared.model_name == 'None' or shared.model is None:
        logger.error("No model is loaded! Select one in the Model tab.")
        yield ''
        return

    # Defining some variables
    cumulative_reply = ''
    prompt = generate_chat_prompt('', state, impersonate=True)
    stopping_strings = get_stopping_strings(state)

    yield text + '...'
    cumulative_reply = text
    for i in range(state['chat_generation_attempts']):
        reply = None
        for reply in generate_reply(prompt + cumulative_reply, state, stopping_strings=stopping_strings, is_chat=True):
            reply = cumulative_reply + reply
            yield reply.lstrip(' ')
            if shared.stop_everything:
                return

        if reply in [None, cumulative_reply]:
            break
        else:
            cumulative_reply = reply

    yield cumulative_reply.lstrip(' ')


def generate_chat_reply(text, state, regenerate=False, _continue=False, loading_message=True):
    history = state['history']
    if regenerate or _continue:
        text = ''
        if (len(history['visible']) == 1 and not history['visible'][0][0]) or len(history['internal']) == 0:
            yield history
            return

    for history in chatbot_wrapper(text, state, regenerate=regenerate, _continue=_continue, loading_message=loading_message):
        yield history


# Same as above but returns HTML for the UI
def generate_chat_reply_wrapper(text, start_with, state, regenerate=False, _continue=False):
    if start_with != '' and not _continue:
        if regenerate:
            text, state['history'] = remove_last_message(state['history'])
            regenerate = False

        _continue = True
        send_dummy_message(text, state)
        send_dummy_reply(start_with, state)

    for i, history in enumerate(generate_chat_reply(text, state, regenerate, _continue, loading_message=True)):
        yield chat_html_wrapper(history, state['name1'], state['name2'], state['mode'], state['chat_style']), history


def remove_last_message(history):
    if len(history['visible']) > 0 and history['internal'][-1][0] != '<|BEGIN-VISIBLE-CHAT|>':
        last = history['visible'].pop()
        history['internal'].pop()
    else:
        last = ['', '']

    return last[0], history


def send_last_reply_to_input(history):
    if len(history['internal']) > 0:
        return history['internal'][-1][1]
    else:
        return ''


def replace_last_reply(text, state):
    history = state['history']
    if len(history['visible']) > 0:
        history['visible'][-1][1] = text
        history['internal'][-1][1] = apply_extensions('input', text, state)

    return history


def send_dummy_message(text, state):
    history = state['history']
    history['visible'].append([text, ''])
    history['internal'].append([apply_extensions('input', text, state), ''])
    return history


def send_dummy_reply(text, state):
    history = state['history']
    if len(history['visible']) > 0 and not history['visible'][-1][1] == '':
        history['visible'].append(['', ''])
        history['internal'].append(['', ''])

    history['visible'][-1][1] = text
    history['internal'][-1][1] = apply_extensions('input', text, state)
    return history


def clear_chat_log(state):
    greeting = state['greeting']
    mode = state['mode']
    history = state['history']

    history['visible'] = []
    history['internal'] = []
    if mode != 'instruct':
        if greeting != '':
            history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', greeting]]
            history['visible'] += [['', apply_extensions('output', greeting, state)]]

    return history


def redraw_html(history, name1, name2, mode, style, reset_cache=False):
    return chat_html_wrapper(history, name1, name2, mode, style, reset_cache=reset_cache)


def save_history(history, path=None):
    p = path or Path('logs/exported_history.json')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(json.dumps(history, indent=4))

    return p


def load_history(file, history):
    try:
        file = file.decode('utf-8')
        j = json.loads(file)
        if 'internal' in j and 'visible' in j:
            return j
        else:
            return history
    except:
        return history


def save_history_at_user_request(history, character, mode):
    def make_timestamp_path(character=None):
        return f"logs/{character or ''}{'_' if character else ''}{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"

    path = None
    if mode in ['chat', 'chat-instruct'] and character not in ['', 'None', None]:
        path = make_timestamp_path(character)
    else:
        # Try to use mode as the file name, otherwise just use the timestamp
        try:
            path = make_timestamp_path(mode.capitalize())
        except:
            path = make_timestamp_path()

    return save_history(history, path)


def save_persistent_history(history, character, mode):
    if mode in ['chat', 'chat-instruct'] and character not in ['', 'None', None] and not shared.args.multi_user:
        save_history(history, path=Path(f'logs/{character}_persistent.json'))


def load_persistent_history(state):
    if state['mode'] == 'instruct':
        return state['history']

    character = state['character_menu']
    greeting = state['greeting']
    p = Path(f'logs/{character}_persistent.json')
    if not shared.args.multi_user and character not in ['None', '', None] and p.exists():
        f = json.loads(open(p, 'rb').read())
        if 'internal' in f and 'visible' in f:
            history = f
        else:
            history = {'internal': [], 'visible': []}
            history['internal'] = f['data']
            history['visible'] = f['data_visible']
    else:
        history = {'internal': [], 'visible': []}
        if greeting != "":
            history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', greeting]]
            history['visible'] += [['', apply_extensions('output', greeting, state)]]

    return history


def replace_character_names(text, name1, name2):
    text = text.replace('{{user}}', name1).replace('{{char}}', name2)
    return text.replace('<USER>', name1).replace('<BOT>', name2)


def build_pygmalion_style_context(data):
    context = ""
    if 'char_persona' in data and data['char_persona'] != '':
        context += f"{data['char_name']}'s Persona: {data['char_persona']}\n"

    if 'world_scenario' in data and data['world_scenario'] != '':
        context += f"Scenario: {data['world_scenario']}\n"

    context = f"{context.strip()}\n<START>\n"
    return context


def generate_pfp_cache(character):
    cache_folder = Path("cache")
    if not cache_folder.exists():
        cache_folder.mkdir()

    for path in [Path(f"characters/{character}.{extension}") for extension in ['png', 'jpg', 'jpeg']]:
        if path.exists():
            img = make_thumbnail(Image.open(path))
            img.save(Path('cache/pfp_character.png'), format='PNG')
            return img

    return None


def load_character(character, name1, name2, instruct=False):
    context = greeting = turn_template = ""
    greeting_field = 'greeting'
    picture = None

    # Deleting the profile picture cache, if any
    if Path("cache/pfp_character.png").exists():
        Path("cache/pfp_character.png").unlink()

    if character not in ['None', '', None]:
        folder = 'characters' if not instruct else 'characters/instruction-following'
        picture = generate_pfp_cache(character)
        filepath = None
        for extension in ["yml", "yaml", "json"]:
            filepath = Path(f'{folder}/{character}.{extension}')
            if filepath.exists():
                break

        if filepath is None:
            logger.error(f"Could not find character file for {character} in {folder} folder. Please check your spelling.")
            return name1, name2, picture, greeting, context, turn_template.replace("\n", r"\n")

        file_contents = open(filepath, 'r', encoding='utf-8').read()
        data = json.loads(file_contents) if extension == "json" else yaml.safe_load(file_contents)

        # Finding the bot's name
        for k in ['name', 'bot', '<|bot|>', 'char_name']:
            if k in data and data[k] != '':
                name2 = data[k]
                break

        # Find the user name (if any)
        for k in ['your_name', 'user', '<|user|>']:
            if k in data and data[k] != '':
                name1 = data[k]
                break

        for field in ['context', 'greeting', 'example_dialogue', 'char_persona', 'char_greeting', 'world_scenario']:
            if field in data:
                data[field] = replace_character_names(data[field], name1, name2)

        if 'context' in data:
            context = data['context']
            if not instruct:
                context = context.strip() + '\n'
        elif "char_persona" in data:
            context = build_pygmalion_style_context(data)
            greeting_field = 'char_greeting'

        if 'example_dialogue' in data:
            context += f"{data['example_dialogue'].strip()}\n"

        if greeting_field in data:
            greeting = data[greeting_field]

        if 'turn_template' in data:
            turn_template = data['turn_template']

    else:
        context = shared.settings['context']
        name2 = shared.settings['name2']
        greeting = shared.settings['greeting']
        turn_template = shared.settings['turn_template']

    return name1, name2, picture, greeting, context, turn_template.replace("\n", r"\n")


@functools.cache
def load_character_memoized(character, name1, name2, instruct=False):
    return load_character(character, name1, name2, instruct=instruct)


def upload_character(json_file, img, tavern=False):
    json_file = json_file if type(json_file) == str else json_file.decode('utf-8')
    data = json.loads(json_file)
    outfile_name = data["char_name"]
    i = 1
    while Path(f'characters/{outfile_name}.json').exists():
        outfile_name = f'{data["char_name"]}_{i:03d}'
        i += 1

    if tavern:
        outfile_name = f'TavernAI-{outfile_name}'

    with open(Path(f'characters/{outfile_name}.json'), 'w', encoding='utf-8') as f:
        f.write(json_file)

    if img is not None:
        img.save(Path(f'characters/{outfile_name}.png'))

    logger.info(f'New character saved to "characters/{outfile_name}.json".')
    return gr.update(value=outfile_name, choices=get_available_characters())


def upload_tavern_character(img, _json):
    _json = {"char_name": _json['name'], "char_persona": _json['description'], "char_greeting": _json["first_mes"], "example_dialogue": _json['mes_example'], "world_scenario": _json['scenario']}
    return upload_character(json.dumps(_json), img, tavern=True)


def check_tavern_character(img):
    if "chara" not in img.info:
        return "Not a TavernAI card", None, None, gr.update(interactive=False)
    decoded_string = base64.b64decode(img.info['chara'])
    _json = json.loads(decoded_string)
    if "data" in _json:
        _json = _json["data"]
    return _json['name'], _json['description'], _json, gr.update(interactive=True)


def upload_your_profile_picture(img):
    cache_folder = Path("cache")
    if not cache_folder.exists():
        cache_folder.mkdir()

    if img is None:
        if Path("cache/pfp_me.png").exists():
            Path("cache/pfp_me.png").unlink()
    else:
        img = make_thumbnail(img)
        img.save(Path('cache/pfp_me.png'))
        logger.info('Profile picture saved to "cache/pfp_me.png"')


def generate_character_yaml(name, greeting, context):
    data = {
        'name': name,
        'greeting': greeting,
        'context': context,
    }

    data = {k: v for k, v in data.items() if v}  # Strip falsy
    return yaml.dump(data, sort_keys=False)


def generate_instruction_template_yaml(user, bot, context, turn_template):
    data = {
        'user': user,
        'bot': bot,
        'turn_template': turn_template,
        'context': context,
    }

    data = {k: v for k, v in data.items() if v}  # Strip falsy
    return yaml.dump(data, sort_keys=False)


def save_character(name, greeting, context, picture, filename):
    if filename == "":
        logger.error("The filename is empty, so the character will not be saved.")
        return

    data = generate_character_yaml(name, greeting, context)
    filepath = Path(f'characters/{filename}.yaml')
    save_file(filepath, data)
    path_to_img = Path(f'characters/{filename}.png')
    if picture is not None:
        picture.save(path_to_img)
        logger.info(f'Saved {path_to_img}.')


def delete_character(name, instruct=False):
    for extension in ["yml", "yaml", "json"]:
        delete_file(Path(f'characters/{name}.{extension}'))

    delete_file(Path(f'characters/{name}.png'))
