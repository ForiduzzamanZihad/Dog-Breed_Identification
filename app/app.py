"""
Main script for gradio app identidog.

The app's main function is defined in the function "run_app"
and relies on three identification functions defined beforehand:
"face_detector", dog_detector" and "breed_identifier".

We use a full frontal Haarcascade model from opencv for face detection
and a pretrained VGG16 CNN from pytorch for dog detection. For breed
identification we load a pretrained ResNet model fine-tuned to 133 dog-breed
categories with fastai (see the app's walkthrough notebook on github
(https://www.github.com/alexeikud/identidog) for the code used in training.)

The final part of the script uses the gradio library
(see https://www.gradio.app/docs/) to launch an iteractive app.

Note: Example images are loaded from the folder "img_examples"
"""

# ## Import used libraries
import os
import pathlib

import cv2
import gradio as gr
import torch
from fastai.vision.all import PILImage, load_learner
from torchvision import models, transforms


# Define function for face detection
def face_detector(img_path: str) -> bool:
    """
    Using opencv's haar cascade classifier to detect human faces in images

    Inputs:
        img_path: path to an image of type string or path object

    Outputs:
        True or False depending on whether at least one face detected or not.
    """
    img = cv2.imread(img_path, 0)  # 0 flag for greyscale
    fd_model = cv2.CascadeClassifier("models/haarcascade_frontalface_alt.xml")
    faces = fd_model.detectMultiScale(img)
    return len(faces) > 0


# load VGG16 model for dog detection
dd_model = models.vgg16(pretrained=True)


# ## Define inference model for dog detection
def dd_predict(img_path: str) -> int:
    """
    Use pre-trained VGG-16 model to obtain index corresponding to
    predicted ImageNet class for image at specified path

    Inputs:
        img_path: path to an image

    Outputs:
        Integer index corresponding to VGG-16 model's prediction
    """
    # Load and pre-process an image from the given img_path
    img = PILImage.create(img_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    img_t = preprocess(img)
    # create a mini-batch
    batch_t = torch.unsqueeze(img_t, 0)
    # initialise model
    dd_model.eval()
    # Use the model and print the predicted category
    probs = dd_model(batch_t).squeeze(0).softmax(0)
    class_id = probs.argmax().item()
    return class_id  # predicted class index


# Set dog category indices from Imagenet1000 classes
DOG_INDICES = range(151, 269)


# Define dog detection function
def dog_detector(img_path: str) -> bool:
    """
    Function returns "True" if a dog is detected in the image.

    Inputs:
        img_path: path to image as str/path.
    Outputs:
        Boolean True/False if dog detected or not.
    """
    image_index = dd_predict(img_path)
    return image_index in DOG_INDICES


# Store path to breed identification model pickle file
MODEL_PATH = pathlib.Path("models/breed_model.pkl")


# Define custom label function saved in fastai learner before loading model
def get_breed_name(filepath: pathlib.Path) -> str:
    """
    Function to grab dog breed name from full pathname.
    The name can be obtained by dropping the last 10
    characters from filename and converting underscores
    to spaces.
    Input:
        filepath as Path object.
    Output:
        breed name as string.
    """
    return filepath.name[:-10].replace("_", " ")


# Load model - we transfer posix to windows path if on windows machine
# (c.f. this thread on Stackoverflow: https://stackoverflow.com/a/68796747)
if os.name == "nt":
    path_temp = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        breeds_inf = load_learner(MODEL_PATH, cpu=True)
    finally:
        pathlib.PosixPath = path_temp
else:
    breeds_inf = load_learner(MODEL_PATH, cpu=True)

# Load breed labels
breeds = breeds_inf.dls.vocab


# Define inference model for breed identification
def breed_identifier(img_path: str) -> dict[str, float]:
    """
    Function to identify dog breed from a possible of 133 categories in the
    udacity dog dataset. It takes an image path, and returns breed and
    probabilities for all breeds. We use a preloaded fastai vision learner
    as a model.
    Input:
        img_path (file path) to image
    Output:
        dictionary whose keys are the identifications categories
        and values are the corresponding probabilities.
    """
    img = PILImage.create(img_path)
    _, _, probs = breeds_inf.predict(img)
    return {breeds[i]: float(probs[i]) for i in range(len(breeds))}


# Defining app main function
def run_app(img_path: str) -> tuple[str, dict[str, float] or None]:
    """
    App predicts most resembling dog breeds from image of human or dog.
    Logic handles cases for a human, dog, both or neither.

    Inputs:
        img_path = path to image
    Outputs:
        detection:  Output display message stored as string
        preds:      Dictionary of prediction confidences with items
                      (breed labels: probabilities).
    """
    # Store boolean values of whether human/dog detected
    face_detected = face_detector(img_path)
    dog_detected = dog_detector(img_path)

    # If both human and dog detected, then output error message
    if face_detected and dog_detected:
        detection = (
            "Both human and dog found! Please crop the image "
            "until only a person or dog are showing to obtain a prediction"
        )
        return detection, None
    # If neither human or dog detected, then output error message
    elif not face_detected and not dog_detected:
        detection = (
            "No human or dog found! Please zoom in by cropping "
            "or upload a new image."
        )
        return detection, None
    # If only a human or a dog detected:
    # Run breed identifier and output relevant message and identification.
    else:
        detection = (
            "Human detected! Your identification results are:"
            if face_detected
            else "Dog detected! The most resembling breeds are:"
        )
        preds = breed_identifier(img_path)
        return detection, preds


# ## Building app with Gradio

# ### Defining Inputs
# app title
title = "Identidog"

# App main description
description = """
Try an example from below, or upload your own image of a dog or a person. \
The app returns the top 3 dog breeds they resemble and the corresponding \
confidences!

**Hint**: Cropped/zoomed-in images showing only a dog or person \
yield the best results!
"""

# App extra info
article = """

 ___

<p style="text-align: center;">
</li>
<a href="https://github.com/alexeikud/identidog"
target="_blank">
Github Repo<a>
</p>
"""
# app inputs
inputs = gr.components.Image(type="filepath", shape=(512, 512), label="image")

# app outputs
outputs = [
    gr.components.Textbox(label="Output:"),
    gr.components.Label(label="Identification results", num_top_classes=3),
]

# input image examples
EXAMPLES_FOLDER = pathlib.Path("img_examples")
examples = [str(img_file) for img_file in EXAMPLES_FOLDER.iterdir()]


# Defining gradio app
app = gr.Interface(
    fn=run_app,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    article=article,
    examples=examples,
    allow_flagging="never",
)

# Launch if script is run directly
if __name__ == "__main__":
    app.launch()
