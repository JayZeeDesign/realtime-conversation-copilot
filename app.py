from flask import Flask, request, jsonify, render_template
import os
import replicate
import tempfile
import requests

app = Flask(__name__)
model = replicate

# here goes your REPLICATE API TOKEN, don't forget to paste it here
os.environ["REPLICATE_API_TOKEN"] = ""


def upload_file(file_path):
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f)}
        r = requests.post('https://file.io/', files=files)

        if r.status_code == 200:
            response_data = r.json()

            return response_data['link']
        else:
            print(f'File upload error: {r.text}')

            return None


@app.route("/")
def index():
    return render_template("index.html")


# function to transcript audio using whisper
@app.route("/process-audio", methods=["POST"])
def process_audio_data():
    audio_data = request.files["audio"].read()
    language = request.form.get('language')

    print("Processing audio in {} language...".format(language))
    # Create a temporary file to save the audio data
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_data)
            temp_audio.flush()

        temp_audio_url = upload_file(temp_audio.name)
        print(temp_audio_url)

        output = replicate.run(
            "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
            input={
                "task": "transcribe",
                "audio": temp_audio_url,
                "language": language,
                "timestamp": "chunk",
                "batch_size": 64,
                "diarise_audio": False
            }
        )

        print(output["text"])

        return jsonify({"transcript": output["text"]})
    except Exception as e:
        print(f"Error running Replicate model: {e}")

        return jsonify({"error": str(e)})


# function to generate suggestion using mixtral
@app.route("/get-suggestion", methods=["POST"])
def get_suggestion():
    data = request.get_json()  # Parse JSON data from the request
    transcript = data.get("transcript", "")  # Extract transcript
    prompt_text = data.get("prompt", "")  # Extract prompt text

    prompt = f"""
    {transcript}
    ------
    {prompt_text}
    """

    print('Sending request for a suggestion with the prompt:')
    print('=====')
    print(prompt)
    print('=====')
    print('Waiting for the API response...')

    suggestion = ""
    for event in model.stream(
        "mistralai/mistral-7b-instruct-v0.2",
        input={
            "debug": False,
            "top_k": 50,
            "top_p": 0.9,
            "prompt": prompt,
            "temperature": 0.6,
            "max_new_tokens": 512,
            "min_new_tokens": -1,
            "prompt_template": "<s>[INST] {prompt} [/INST] ",
            "repetition_penalty": 1.15,
        },
    ):
        suggestion_piece = str(event)
        print('Output: ' + suggestion_piece)
        suggestion += suggestion_piece  # Accumulate the output

    print(suggestion)

    return jsonify({"suggestion": suggestion})  # Send as JSON response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
