# The GPT web UI as a template based Flask app
import os
import requests
import json
import asyncio
import re
import logging
import sys

from collections import deque

from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory, Response
from dotenv import load_dotenv

from gpt_code_ui.kernel_program.main import APP_PORT as KERNEL_APP_PORT

load_dotenv(".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")

UPLOAD_FOLDER = "workspace/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

APP_PORT = int(os.environ.get("WEB_PORT", 8080))


# A helper class to store a limited-length string
class LimitedLengthString:
    def __init__(self, maxlen=2000):
        self.data = deque()  # Use deque to efficiently append and remove elements
        self.len = 0  # Current length of the stored string
        self.maxlen = maxlen  # Maximum allowed length

    def append(self, string):
        self.data.append(string)  # Append the string to the deque
        self.len += len(string)  # Update the current length
        while self.len > self.maxlen:
            # If the length exceeds the maximum allowed length,
            # remove elements from the left
            popped = self.data.popleft()
            self.len -= len(popped)

    def get_string(self):
        # Concatenate all elements in the deque and return the result
        result = "".join(self.data)
        return result[-self.maxlen :]  # Return the last maxlen characters


# Create an instance of the LimitedLengthString class to store message history
message_buffer = LimitedLengthString()


def allowed_file(filename):
    # Placeholder function to determine if a file is allowed for upload
    return True


async def get_code(user_prompt, user_openai_key=None, model="gpt-4"):
    # Asynchronously generate code based on the user prompt using the OpenAI API

    # Construct the prompt for the OpenAI API by combining
    # user prompt and message history
    prompt = f"""    
First, here are my prior requests for context. 
    
<history>
{message_buffer.get_string()}
</history>

Now, your task is to write Python code that accomplishes the following task: 

<task>
{user_prompt}
</task>

Your code will be executed in a Jupyter Python kernel, so please ensure that your 
code will run without errors in this environment. 

Additionally, please note that your response should only be the code itself. 
Do not include any other outputs in your response. 

If you would like to provide a download link, please print it as the following HTML: 
<a href='/download?file=INSERT_FILENAME_HERE'>Download file</a>, 
replacing INSERT_FILENAME_HERE with the actual filename. 

Please provide your python scode below, ensuring that it is clear and easy to read.
"""

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.7,
    }

    final_openai_key = OPENAI_API_KEY
    if user_openai_key:
        final_openai_key = user_openai_key

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {final_openai_key}",
    }

    # Send a POST request to the OpenAI API to get code completions
    response = requests.post(
        f"{OPENAI_BASE_URL}/v1/chat/completions",
        data=json.dumps(data),
        headers=headers,
    )

    def extract_code(text):
        # Extract code from the response text using regular expressions
        # Match triple backtick blocks first
        triple_match = re.search(r"```(?:\w+\n)?(.+?)```", text, re.DOTALL)
        if triple_match:
            return triple_match.group(1).strip()
        else:
            # If no triple backtick blocks, match single backtick blocks
            single_match = re.search(r"`(.+?)`", text, re.DOTALL)
            if single_match:
                return single_match.group(1).strip()
        # If no code blocks found, return the original text
        return text

    if response.status_code != 200:
        return "Error: " + response.text, 500

    # Extract the generated code from the response JSON
    generated_code = extract_code(response.json()["choices"][0]["message"]["content"])
    return generated_code, 200


# Disable the Werkzeug logger
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

# Disable the Flask server banner
cli = sys.modules["flask.cli"]
cli.show_server_banner = lambda *x: None

# Initialize the Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)


@app.route("/")
def index():
    # Serve the index.html file from the static folder
    if not os.path.exists(os.path.join(app.root_path, "static/index.html")):
        print(
            "index.html not found in static folder."
            "Exiting."
            "Did you forget to run `make compile_frontend`"
            "before installing the local package?"
        )

    return send_from_directory("static", "index.html")


@app.route("/api/<path:path>", methods=["GET", "POST"])
def proxy_kernel_manager(path):
    # Proxy requests to the Jupyter Python kernel manager
    if request.method == "POST":
        resp = requests.post(
            f"http://localhost:{KERNEL_APP_PORT}/{path}", json=request.get_json()
        )
    else:
        resp = requests.get(f"http://localhost:{KERNEL_APP_PORT}/{path}")

    excluded_headers = [
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "connection",
    ]
    headers = [
        (name, value)
        for (name, value) in resp.raw.headers.items()
        if name.lower() not in excluded_headers
    ]

    response = Response(resp.content, resp.status_code, headers)
    return response


@app.route("/assets/<path:path>")
def serve_static(path):
    # Serve static files from the static/assets/ directory
    return send_from_directory("static/assets/", path)


@app.route("/download")
def download_file():
    # Download a file from the workspace/ directory

    # Get the file name from the query parameter
    file = request.args.get("file")

    # Send the file from the workspace/ directory with appropriate headers
    # to trigger download
    return send_from_directory(
        os.path.join(os.getcwd(), "workspace"), file, as_attachment=True
    )


@app.route("/inject-context", methods=["POST"])
def inject_context():
    # Receive a user prompt and append it to the message buffer
    user_prompt = request.json.get("prompt", "")
    message_buffer.append(user_prompt + "\n\n")
    return jsonify({"result": "success"})


@app.route("/generate", methods=["POST"])
def generate_code():
    # Generate Python code based on the user prompt

    # Get user prompt, OpenAI API key, and model name from the JSON payload
    user_prompt = request.json.get("prompt", "")
    user_openai_key = request.json.get("openAIKey", None)
    model = request.json.get("model", None)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Call the get_code function asynchronously to generate code
    generated_code, status = loop.run_until_complete(
        get_code(user_prompt, user_openai_key, model)
    )
    loop.close()

    # Append the user prompt to the message buffer
    message_buffer.append(user_prompt + "\n\n")

    # Return the generated code and status in a JSON response
    return jsonify({"code": generated_code}), status


@app.route("/upload", methods=["POST"])
def upload_file():
    # Handle file uploads

    # Check if the file part exists in the request
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    # Check if a file was selected
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Check if the file type is allowed (placeholder function)
    if file and allowed_file(file.filename):
        # Save the file to the UPLOAD_FOLDER
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], file.filename))
        return jsonify({"message": "File successfully uploaded"}), 200
    else:
        return jsonify({"error": "File type not allowed"}), 400


if __name__ == "__main__":
    # Run the Flask app on 0.0.0.0 (all available network interfaces) and the specified APP_PORT
    app.run(host="0.0.0.0", port=APP_PORT, debug=True, use_reloader=False)
