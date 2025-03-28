from flask import Flask, request, jsonify, render_template
import base64
from io import BytesIO
from diffusers import StableDiffusionPipeline
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu" #macbook air intel 2017 likely use cpu
pipe = pipe.to(device)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result="Waiting for prompt")

@app.route("/generate", methods=["POST"])
def generate_art():
    data = request.get_json()
    prompt = data["prompt"]

    image = pipe(prompt).images[0]

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({"image": img_str})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    
