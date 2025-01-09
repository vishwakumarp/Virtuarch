# VirtuArch

This repository demonstrates the integration of Stable Diffusion pipelines, including ControlNet and Stable Diffusion XL, into a Flask web application. The application enables users to upload images and generate AI-powered image transformations for various use cases, including:

- Architectural and interior design enhancements.
- Creative sketch refinement and materialization.
- Style-guided image generation.

---

## Features

### 1. **Image Upload and Management**
- Users can upload their initial sketches or images.
- Automatic naming and management of uploaded and generated images.

### 2. **AI-Powered Image Processing Pipelines**
- **ControlNet Pipeline**:
  - Enhances raw architectural and interior design sketches using Canny edge detection and fine-tuned hyperparameters.
- **Stable Diffusion XL**:
  - Transforms images with style-guided refinement.
  - Supports advanced image-to-image and text-to-image generation workflows.

### 3. **Interactive Web Interface**
- Multiple pages for different pipelines:
  - **Page 1**: ControlNet transformations.
  - **Page 2**: Stable Diffusion XL with refinements.
  - **Page 3**: Standard Stable Diffusion XL transformations.

### 4. **GPU Acceleration**
- Full CUDA support for high-performance image processing.

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/<your-username>/stable-diffusion-flask.git
cd stable-diffusion-flask
```

2. **Set up a Python environment:**
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download pre-trained models:**
- Ensure you have access to the required models from Hugging Face.
- Place the models in appropriate directories as needed.

---

## Usage

1. **Run the Flask application:**
```bash
python app.py
```

2. **Access the web application:**
- Navigate to `http://127.0.0.1:5000` in your browser.

3. **Upload and transform images:**
- Choose the appropriate page based on the pipeline:
  - **Page 1:** Sketch to materialized product with ControlNet.
  - **Page 2:** Style-guided transformations with Stable Diffusion XL.
  - **Page 3:** General image-to-image processing.

---

## Example Use Case

### Leveraging AI for Design Innovation:
- **Sketch Refinement:**
  - Convert raw sketches into detailed architectural designs using ControlNet.
- **Materialization:**
  - Apply different iterations of Stable Diffusion to visualize design concepts, significantly boosting client engagement.

---

## Key Technologies
- **Python & Flask:** Web application framework.
- **PyTorch & Diffusers:** For running Stable Diffusion models.
- **OpenCV & NumPy:** Image preprocessing and manipulation.
- **Pillow (PIL):** For image file handling.

---

## Project Structure
```
.
├── app.py                # Main Flask application
├── static/
│   ├── uploads/          # Uploaded images
│   └── generated_images/ # AI-generated images
├── templates/
│   ├── home.html         # Home page template
│   ├── page1.html        # ControlNet page
│   ├── page2.html        # Stable Diffusion XL page
│   ├── page3.html        # General transformations
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contribution
Contributions are welcome! Please submit issues or pull requests to improve the functionality or documentation.

---

## Acknowledgments
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index): For pre-trained model pipelines.
- [PyTorch](https://pytorch.org/): For deep learning framework.
- [ControlNet](https://github.com/lllyasviel/ControlNet): For sketch-to-image transformation.
