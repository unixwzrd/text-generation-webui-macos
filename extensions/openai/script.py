import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from modules import shared

from extensions.openai.tokens import token_count, token_encode, token_decode
import extensions.openai.models as OAImodels
import extensions.openai.edits as OAIedits
import extensions.openai.embeddings as OAIembeddings
import extensions.openai.images as OAIimages
import extensions.openai.moderations as OAImoderations
import extensions.openai.completions as OAIcompletions
from extensions.openai.errors import *
from extensions.openai.utils import debug_msg
from extensions.openai.defaults import (get_default_req_params, default, clamp)


params = {
    'port': int(os.environ.get('OPENEDAI_PORT')) if 'OPENEDAI_PORT' in os.environ else 5001,
}


class Handler(BaseHTTPRequestHandler):
    def send_access_control_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET,HEAD,OPTIONS,POST,PUT"
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Origin, Accept, X-Requested-With, Content-Type, "
            "Access-Control-Request-Method, Access-Control-Request-Headers, "
            "Authorization"
        )

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_access_control_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write("OK".encode('utf-8'))

    def start_sse(self):
        self.send_response(200)
        self.send_access_control_headers()
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        # self.send_header('Connection', 'keep-alive')
        self.end_headers()

    def send_sse(self, chunk: dict):
        response = 'data: ' + json.dumps(chunk) + '\r\n\r\n'
        debug_msg(response[:-4])
        self.wfile.write(response.encode('utf-8'))

    def end_sse(self):
        response = 'data: [DONE]\r\n\r\n'
        debug_msg(response[:-4])
        self.wfile.write(response.encode('utf-8'))

    def return_json(self, ret: dict, code: int = 200, no_debug=False):
        self.send_response(code)
        self.send_access_control_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

        response = json.dumps(ret)
        r_utf8 = response.encode('utf-8')
        self.wfile.write(r_utf8)
        if not no_debug:
            debug_msg(r_utf8)

    def openai_error(self, message, code=500, error_type='APIError', param='', internal_message=''):

        error_resp = {
            'error': {
                'message': message,
                'code': code,
                'type': error_type,
                'param': param,
            }
        }
        if internal_message:
            print(error_type, message)
            print(internal_message)
            # error_resp['internal_message'] = internal_message

        self.return_json(error_resp, code)

    def openai_error_handler(func):
        def wrapper(self):
            try:
                func(self)
            except InvalidRequestError as e:
                self.openai_error(e.message, e.code, e.__class__.__name__, e.param, internal_message=e.internal_message)
            except OpenAIError as e:
                self.openai_error(e.message, e.code, e.__class__.__name__, internal_message=e.internal_message)
            except Exception as e:
                self.openai_error(repr(e), 500, 'OpenAIError', internal_message=traceback.format_exc())

        return wrapper

    @openai_error_handler
    def do_GET(self):
        debug_msg(self.requestline)
        debug_msg(self.headers)

        if self.path.startswith('/v1/engines') or self.path.startswith('/v1/models'):
            is_legacy = 'engines' in self.path
            is_list = self.path in ['/v1/engines', '/v1/models']
            if is_legacy and not is_list:
                model_name = self.path[self.path.find('/v1/engines/') + len('/v1/engines/'):]
                resp = OAImodels.load_model(model_name)
            elif is_list:
                resp = OAImodels.list_models(is_legacy)
            else:
                model_name = self.path[len('/v1/models/'):]
                resp = OAImodels.model_info()

            self.return_json(resp)

        elif '/billing/usage' in self.path:
            #  Ex. /v1/dashboard/billing/usage?start_date=2023-05-01&end_date=2023-05-31
            self.return_json({"total_usage": 0}, no_debug=True)

        else:
            self.send_error(404)

    @openai_error_handler
    def do_POST(self):
        debug_msg(self.requestline)
        debug_msg(self.headers)

        content_length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(content_length).decode('utf-8'))

        debug_msg(body)

        if '/completions' in self.path or '/generate' in self.path:

            if not shared.model:
                raise ServiceUnavailableError("No model loaded.")

            is_legacy = '/generate' in self.path
            is_streaming = body.get('stream', False)

            if is_streaming:
                self.start_sse()

                response = []
                if 'chat' in self.path:
                    response = OAIcompletions.stream_chat_completions(body, is_legacy=is_legacy)
                else:
                    response = OAIcompletions.stream_completions(body, is_legacy=is_legacy)

                for resp in response:
                    self.send_sse(resp)

                self.end_sse()

            else:
                response = ''
                if 'chat' in self.path:
                    response = OAIcompletions.chat_completions(body, is_legacy=is_legacy)
                else:
                    response = OAIcompletions.completions(body, is_legacy=is_legacy)

                self.return_json(response)

        elif '/edits' in self.path:
            # deprecated

            if not shared.model:
                raise ServiceUnavailableError("No model loaded.")

            req_params = get_default_req_params()

            instruction = body['instruction']
            input = body.get('input', '')
            temperature = clamp(default(body, 'temperature', req_params['temperature']), 0.001, 1.999)  # fixup absolute 0.0
            top_p = clamp(default(body, 'top_p', req_params['top_p']), 0.001, 1.0)

            response = OAIedits.edits(instruction, input, temperature, top_p)

            self.return_json(response)

        elif '/images/generations' in self.path:
            if not 'SD_WEBUI_URL' in os.environ:
                raise ServiceUnavailableError("Stable Diffusion not available. SD_WEBUI_URL not set.")

            prompt = body['prompt']
            size = default(body, 'size', '1024x1024')
            response_format = default(body, 'response_format', 'url')  # or b64_json
            n = default(body, 'n', 1)  # ignore the batch limits of max 10

            response = OAIimages.generations(prompt=prompt, size=size, response_format=response_format, n=n)

            self.return_json(response, no_debug=True)

        elif '/embeddings' in self.path:
            encoding_format = body.get('encoding_format', '')

            input = body.get('input', body.get('text', ''))
            if not input:
                raise InvalidRequestError("Missing required argument input", params='input')

            if type(input) is str:
                input = [input]

            response = OAIembeddings.embeddings(input, encoding_format)

            self.return_json(response, no_debug=True)

        elif '/moderations' in self.path:
            input = body['input']
            if not input:
                raise InvalidRequestError("Missing required argument input", params='input')

            response = OAImoderations.moderations(input)

            self.return_json(response, no_debug=True)

        elif self.path == '/api/v1/token-count':
            # NOT STANDARD. lifted from the api extension, but it's still very useful to calculate tokenized length client side.
            response = token_count(body['prompt'])

            self.return_json(response, no_debug=True)

        elif self.path == '/api/v1/token/encode':
            # NOT STANDARD. needed to support logit_bias, logprobs and token arrays for native models
            encoding_format = body.get('encoding_format', '')

            response = token_encode(body['input'], encoding_format)

            self.return_json(response, no_debug=True)

        elif self.path == '/api/v1/token/decode':
            # NOT STANDARD. needed to support logit_bias, logprobs and token arrays for native models
            encoding_format = body.get('encoding_format', '')

            response = token_decode(body['input'], encoding_format)

            self.return_json(response, no_debug=True)

        else:
            self.send_error(404)


def run_server():
    server_addr = ('0.0.0.0' if shared.args.listen else '127.0.0.1', params['port'])
    server = ThreadingHTTPServer(server_addr, Handler)
    if shared.args.share:
        try:
            from flask_cloudflared import _run_cloudflared
            public_url = _run_cloudflared(params['port'], params['port'] + 1)
            print(f'OpenAI compatible API ready at: OPENAI_API_BASE={public_url}/v1')
        except ImportError:
            print('You should install flask_cloudflared manually')
    else:
        print(f'OpenAI compatible API ready at: OPENAI_API_BASE=http://{server_addr[0]}:{server_addr[1]}/v1')

    server.serve_forever()


def setup():
    Thread(target=run_server, daemon=True).start()
