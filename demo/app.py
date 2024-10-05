import gradio as gr
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d

# Constants
MODEL_ID = "stabilityai/stable-diffusion-2-1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
def load_model(model_id):
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    return pipeline.to(DEVICE)

pip_2_1 = load_model(MODEL_ID)

# Global variables for caching
prompt_prev, sd_options_prev, seed_prev, sd_image_prev = None, None, None, None

def infer(prompt, sd_options, seed, b1, b2, s1, s2):
    global prompt_prev, sd_options_prev, seed_prev, sd_image_prev

    run_baseline = prompt != prompt_prev or sd_options != sd_options_prev or seed != seed_prev
    if run_baseline:
        prompt_prev, sd_options_prev, seed_prev = prompt, sd_options, seed
        register_free_upblock2d(pip_2_1, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
        register_free_crossattn_upblock2d(pip_2_1, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
        
        torch.manual_seed(seed)
        print("Generating SD:")
        sd_image = pip_2_1(prompt, num_inference_steps=25).images[0]
        sd_image_prev = sd_image
    else:
        sd_image = sd_image_prev

    register_free_upblock2d(pip_2_1, b1=b1, b2=b2, s1=s1, s2=s1)
    register_free_crossattn_upblock2d(pip_2_1, b1=b1, b2=b2, s1=s1, s2=s1)

    torch.manual_seed(seed)
    print("Generating FreeU:")
    freeu_image = pip_2_1(prompt, num_inference_steps=25).images[0]

    return [sd_image, freeu_image]

# Example prompts
examples = [
    ["A drone view of celebration with Christmas tree and fireworks, starry sky - background."],
    ["happy dog wearing a yellow turtleneck, studio, portrait, facing camera, studio, dark bg"],
    # ... (other examples)
]

# Gradio interface
def create_interface():
    with gr.Blocks(css='style.css') as block:
        gr.Markdown("# SD 2.1 vs. FreeU")
        with gr.Group():
            with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                text = gr.Textbox(label="Enter your prompt", show_label=False, max_lines=1, 
                                  placeholder="Enter your prompt", container=False)
                btn = gr.Button("Generate image", scale=0)
            sd_options = gr.Dropdown(["SD2.1"], label="SD options", value="SD2.1", visible=False)

        with gr.Group():
            with gr.Accordion('FreeU Parameters', open=False):
                b1 = gr.Slider(label='b1: backbone factor of the first stage block of decoder', 
                               minimum=1, maximum=1.6, step=0.01, value=1.1)
                b2 = gr.Slider(label='b2: backbone factor of the second stage block of decoder', 
                               minimum=1, maximum=1.6, step=0.01, value=1.2)
                s1 = gr.Slider(label='s1: skip factor of the first stage block of decoder', 
                               minimum=0, maximum=1, step=0.1, value=0.2)
                s2 = gr.Slider(label='s2: skip factor of the second stage block of decoder', 
                               minimum=0, maximum=1, step=0.1, value=0.2)
                seed = gr.Slider(label='seed', minimum=0, maximum=1000, step=1, value=42)

        with gr.Row():
            with gr.Column():
                image_1 = gr.Image(interactive=False)
                gr.Markdown("SD")
            with gr.Column():
                image_2 = gr.Image(interactive=False)
                gr.Markdown("FreeU")

        inputs = [text, sd_options, seed, b1, b2, s1, s2]
        outputs = [image_1, image_2]

        gr.Examples(examples=examples, fn=infer, inputs=inputs, outputs=outputs, cache_examples=False)

        text.submit(infer, inputs=inputs, outputs=outputs)
        btn.click(infer, inputs=inputs, outputs=outputs)

    return block

# Launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
