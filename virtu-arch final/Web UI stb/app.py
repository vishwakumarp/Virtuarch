from distributed import UploadFile
from flask import Flask, jsonify, request, render_template, redirect, url_for, send_file
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image, AutoPipelineForImage2Image, DiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from datetime import datetime
from diffusers.utils import load_image, make_image_grid
import cv2
import numpy as np
from torchvision.transforms.functional import to_pil_image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
GENERATED_IMAGES_FOLDER = 'static/generated_images'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_IMAGES_FOLDER'] = GENERATED_IMAGES_FOLDER


def get_next_uploaded_image_name(folder):
    count = 1
    while True:
        image_name = f"img_{count:03d}.jpg"
        if not os.path.exists(os.path.join(folder, image_name)):
            return image_name
        count += 1

def get_next_generated_image_name(folder):
    count = 1
    while True:
        image_name = f"gen_img_{count:03d}.jpg"
        if not os.path.exists(os.path.join(folder, image_name)):
            pathname(image_name)
            return image_name
        count += 1

def pathname(image_name):
    img = image_name
    gen_path = '../static/'+image_name
    print(gen_path)
    return render_template('page1.html', gen_path=gen_path)





@app.route('/')
def home():
    return render_template('home.html')
uploaded_image_name = ""
generated_image_name = ""
# @app.route('/page1')
# def page1():
#     return render_template('templates\page1.html')
# In your Flask route for page1
@app.route('/page1')
def page1():

    return render_template('page1.html', page_name='page1')

# In your Flask route for page2
@app.route('/page2')
def page2():
    return render_template('page2.html', page_name='page2')
# In your Flask route for page3
@app.route('/page3')
def page3():
    return render_template('page3.html', page_name='page3')


@app.route('/submit', methods=['POST'])
def handle_form_submission():
    global pipeline, refiner
    # Extract common form data
    source_page = request.form.get('sourcePage')  # Get the source page identifier
    print(source_page)

    if not source_page:
        return jsonify({"error": "sourcePage field is missing or empty"}), 400

    if not torch.cuda.is_available():
        print("CUDA is not available. Ensure you have installed the required drivers.")
    else:
        print("CUDA is available. Proceeding with GPU support.")
    style_description = request.form['styleDescription']
    strength = float(request.form['strength'])
    guidance_scale = float(request.form['guidanceScale'])
    num_inference_steps = int(request.form['numInferenceSteps'])
    source_page = request.form.get('sourcePage')  # Get the source page identifier
    print(source_page)
    # Handle file upload
    uploaded_file = request.files['imageUpload']
    uploaded_image_name = get_next_uploaded_image_name(app.config['UPLOAD_FOLDER'])
    uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image_name))
    uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image_name)
    init_image = Image.open(uploaded_image_path).convert("RGB")
    torch.cuda.empty_cache()
    # Image processing logic based on the source page
    if source_page == 'page1':
        #PIPELINE FOR CONTROL-NET
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        control_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        )
        control_pipe.scheduler = UniPCMultistepScheduler.from_config(control_pipe.scheduler.config)
        control_pipe.to("cuda")
        # Specific processing for page1
        image_mod = np.array(init_image)
        low_threshold = 100
        high_threshold = 200
        image_mod = cv2.Canny(image_mod, low_threshold, high_threshold)
        image_mod = image_mod[:, :, None]
        image_mod = np.concatenate([image_mod, image_mod, image_mod], axis=2)
        canny_image = Image.fromarray(image_mod)
        images = control_pipe(prompt=style_description,image = canny_image,strength=strength, guidance_scale=guidance_scale,negative_prompt="blurry, blurred, lowres, monochrome, bad anatomy, worst quality, low quality",num_inference_steps=num_inference_steps).images

        generated_image_name = get_next_generated_image_name(app.config['GENERATED_IMAGES_FOLDER'])
        generated_image_path = os.path.join(app.config['GENERATED_IMAGES_FOLDER'], generated_image_name)
        images[0].save(generated_image_path)


        # uploaded_image_path = url_for('static', filename=f'uploads/{uploaded_image_name}')
        # print("uploaded_image_path")
        # generated_image_path = url_for('static', filename=f'generated_images/{generated_image_name}')
        print('Uploaded path: ',uploaded_image_path)
        print('Uploaded name: ',uploaded_image_name)
        print('Generated path: ',generated_image_path)
        print('Generated name: ',generated_image_name)
        return jsonify({
        "uploaded_image_name": uploaded_image_name,
        "generated_image_name": generated_image_name })

        #return render_template('page1.html', uploaded_image_name=uploaded_image_name, generated_image_name=generated_image_name)
        # return redirect(url_for('page1', uploaded_image_name=uploaded_image_name, generated_image_name=generated_image_name))

    elif source_page == 'page2':
    #PIPELINE FOR STABLE DIFFUSION XL + REFINER
        pipeline_text2image = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")

        refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipeline_text2image.text_encoder_2,
        vae=pipeline_text2image.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        ).to("cuda")

        pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to("cuda")

        images = pipeline(prompt=style_description, image=init_image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, scheduler='DDIM',output_type="latent").images[0]
        images = refiner(prompt=style_description, image=images[None, :]).images

        generated_image_name = get_next_generated_image_name(app.config['GENERATED_IMAGES_FOLDER'])
        generated_image_path = os.path.join(app.config['GENERATED_IMAGES_FOLDER'], generated_image_name)
        images[0].save(generated_image_path)
        # image_pil = to_pil_image(images[0])
        # generated_image_name = get_next_generated_image_name(app.config['GENERATED_IMAGES_FOLDER'])
        # generated_image_path = os.path.join(app.config['GENERATED_IMAGES_FOLDER'], generated_image_name)
        # image_pil.save(generated_image_path)

        return jsonify({
        "uploaded_image_name": uploaded_image_name,
        "generated_image_name": generated_image_name })
        # uploaded_image_path = url_for('static', filename=f'uploads/{uploaded_image_name}')
        # generated_image_path = url_for('static', filename=f'generated_images/{generated_image_name}')
        # generated_image_name = get_next_generated_image_name(app.config['GENERATED_IMAGES_FOLDER'])
        # generated_image_path = os.path.join(app.config['GENERATED_IMAGES_FOLDER'], generated_image_name)
        # images[0].save(generated_image_path)
        # return render_template('page2.html', uploaded_image_path=uploaded_image_path, generated_image_path=generated_image_path)

    elif source_page == 'page3':
        #Stable Diffusion XL
        #pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        #pipe.to("cuda")
        pipeline_text2image = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")
        pipe = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to("cuda")
        images = pipe(prompt=style_description, image=init_image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, scheduler='DDIM').images

        generated_image_name = get_next_generated_image_name(app.config['GENERATED_IMAGES_FOLDER'])
        generated_image_path = os.path.join(app.config['GENERATED_IMAGES_FOLDER'], generated_image_name)
        images[0].save(generated_image_path)
        # images[0].save(generated_image_path)
        # uploaded_image_path = url_for('static', filename=f'uploads/{uploaded_image_name}')
        # generated_image_path = url_for('static', filename=f'generated_images/{generated_image_name}')
        # generated_image_name = get_next_generated_image_name(app.config['GENERATED_IMAGES_FOLDER'])
        # generated_image_path = os.path.join(app.config['GENERATED_IMAGES_FOLDER'], generated_image_name)
        # images[0].save(generated_image_path)
        # return render_template('page3.html', uploaded_image_path=uploaded_image_path, generated_image_path=generated_image_path)
        # image_pil = to_pil_image(images[0])
        # generated_image_name = get_next_generated_image_name(app.config['GENERATED_IMAGES_FOLDER'])
        # generated_image_path = os.path.join(app.config['GENERATED_IMAGES_FOLDER'], generated_image_name)
        # image_pil.save(generated_image_path)

        return jsonify({
        "uploaded_image_name": uploaded_image_name,
        "generated_image_name": generated_image_name })

# @app.route('/index')
# def result():
#     #GET THE IMAGE FILES FOR RESULT DISPLAYING
#     uploaded_image_name = request.args.get('uploaded_image_name', '')
#     generated_image_name = request.args.get('generated_image_name', '')

#     #IMAGE_PATHS
#     uploaded_image_path = url_for('static', filename=f'uploads/{uploaded_image_name}')
#     generated_image_path = url_for('static', filename=f'generated_images/{generated_image_name}')
#     return render_template('index.html', uploaded_image_path=uploaded_image_path, generated_image_path=generated_image_path)

if __name__ == '__main__':
    # Ensure the 'uploads' and 'executed_notebooks' directories exist
    # os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # os.makedirs('executed_notebooks', exist_ok=True)

    app.run(debug=False)
