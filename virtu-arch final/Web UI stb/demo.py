from flask import Flask, request, render_template
import os
import base64
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def handle_form_submission():
    # Extract form data
    style_description = request.form['styleDescription']
    strength = float(request.form['strength'])
    guidance_scale = float(request.form['guidanceScale'])
    num_inference_steps = int(request.form['numInferenceSteps'])

    # Handle file upload
    uploaded_file = request.files['imageUpload']
    
    # Save the uploaded file
    uploaded_file_path = os.path.join('uploads', uploaded_file.filename)
    uploaded_file.save(uploaded_file_path)

    if torch.cuda.is_available():
        device = "cuda"
        print("GPU is available!")
    else:
        print("GPU is not available. The code will run on CPU.")
        device = "cpu"

    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    init_image = Image.open(uploaded_file_path).convert("RGB")

    # Set the prompt
    prompt = style_description

    # Generate the image
    images = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, scheduler='K-EULER').images

    # Save the generated image
    generated_image_path = os.path.join('static', 'generated_images', 'output.png')
    images[0].save(generated_image_path)

    # Encode the image as base64
    with open(generated_image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Render the template with the base64-encoded image
    return render_template('result.html', generated_image=encoded_image)

if __name__ == '__main__':
    # Ensure the 'uploads' and 'static/generated_images' directories exist
    os.makedirs('uploads', exist_ok=True)
    os.makedirs(os.path.join('static', 'generated_images'), exist_ok=True)

    app.run(debug=True)
