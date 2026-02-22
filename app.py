import torch
import gradio as gr
from torchvision import transforms
from configs.config import Config
from models.multitask_model import MultiTaskModel
from utils.inference import load_model, predict
from utils.gradcam import GradCAM, overlay_heatmap

config = Config()
model = load_model(MultiTaskModel, config.MODEL_SAVE_PATH, config.DEVICE)

transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


def inference_with_gradcam(image, explain_target):

    results = predict(model, image, config.DEVICE)
    input_tensor = transform(image).unsqueeze(0).to(config.DEVICE)

    # Dynamically select correct layer
    if explain_target == "Gender":
        target_layer = model.gender_adapter.block[-1]
        class_idx = 0
    else:
        target_layer = model.smile_adapter.block[-1]
        class_idx = 1

    gradcam = GradCAM(model, target_layer)

    cam = gradcam.generate(input_tensor, class_idx)
    overlay = overlay_heatmap(
        image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        cam
    )

    return (
        results["gender_label"],
        results["gender_conf"],
        results["smile_label"],
        results["smile_conf"],
        overlay
    )


with gr.Blocks(title="Facial Attribute Multitask System") as demo:

    gr.Markdown("## Facial Attribute Multitask System")
    gr.Markdown( "Predict gender and smile attributes using a spatially decoupled multi-task CNN "
        "with real-time Grad-CAM explainability.")

    with gr.Row():

        with gr.Column():
            image_input = gr.Image(type="pil", label="Input Image")
            explain_choice = gr.Radio(
                ["Gender", "Smile"],
                value="Smile",
                label="Explain Prediction For"
            )
            predict_button = gr.Button("Predict")

        with gr.Column():
            gr.Markdown("### Predictions")

            gender_output = gr.Textbox(label="Gender")
            gender_conf = gr.Slider(0, 100, step=0.1,
                                    label="Gender Confidence (%)",
                                    interactive=False)

            smile_output = gr.Textbox(label="Smile")
            smile_conf = gr.Slider(0, 100, step=0.1,
                                   label="Smile Confidence (%)",
                                   interactive=False)

            gr.Markdown("### Grad-CAM Visualization")
            cam_output = gr.Image(label="Attention Map")

    predict_button.click(
        fn=inference_with_gradcam,
        inputs=[image_input, explain_choice],
        outputs=[
            gender_output,
            gender_conf,
            smile_output,
            smile_conf,
            cam_output
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)