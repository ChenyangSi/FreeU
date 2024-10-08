import gradio as gr
from PIL import Image  
import torch
from diffusers import StableDiffusionPipeline
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d

# Load the model
model_id = "stabilityai/stable-diffusion-2-1"
pip_2_1 = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pip_2_1 = pip_2_1.to("cuda")

# Global variables for caching previous results
prompt_prev = None
sd_options_prev = None
seed_prev = None 
sd_image_prev = None

# Inference function
def infer(prompt, sd_options, seed, b1, b2, s1, s2):
    global prompt_prev, sd_options_prev, seed_prev, sd_image_prev

    pip = pip_2_1
    run_baseline = (prompt != prompt_prev or sd_options != sd_options_prev or seed != seed_prev)

    # Update cached inputs
    if run_baseline:
        prompt_prev = prompt
        sd_options_prev = sd_options
        seed_prev = seed

        # Initial baseline SD image generation
        register_free_upblock2d(pip, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
        register_free_crossattn_upblock2d(pip, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
        torch.manual_seed(seed)
        print("Generating SD:")
        sd_image = pip(prompt, num_inference_steps=25).images[0]  
        sd_image_prev = sd_image
    else:
        sd_image = sd_image_prev

    # FreeU image generation with adjustable parameters
    register_free_upblock2d(pip, b1=b1, b2=b2, s1=s1, s2=s2)
    register_free_crossattn_upblock2d(pip, b1=b1, b2=b2, s1=s1, s2=s2)
    torch.manual_seed(seed)
    print("Generating FreeU:")
    freeu_image = pip(prompt, num_inference_steps=25).images[0]  

    return [sd_image, freeu_image]

# Example prompts
examples = [
    ["A drone view of celebration with a Christmas tree and fireworks, starry sky."],
    ["Happy dog wearing a yellow turtleneck, studio portrait, dark background."],
    ["Campfire at night in a snowy forest with a starry sky."],
    ["A fantasy landscape, trending on ArtStation."],
    ["Busy freeway at night."],
    ["An astronaut riding a horse in space, photorealistic."],
    ["Turtle swimming in the ocean."],
    ["A stormtrooper vacuuming the beach."],
    ["An astronaut feeding ducks by a lake, reflections on water."],
    ["Fireworks."],
    ["A fat rabbit in a purple robe walking through a fantasy landscape."],
    ["A koala bear playing piano in the forest."],
    ["An astronaut flying in space, 4k resolution."],
    ["Flying through fantasy landscapes, 4k resolution."],
    ["A cabin on a snowy mountain in Disney style, ArtStation."],
    ["Half-human, half-cat hybrid."],
    ["A drone flying over a snowy forest."],
]

# CSS styling
css = """
h1 {
  text-align: center;
}

#component-0 {
  max-width: 730px;
  margin: auto;
}
"""

# Building the Gradio UI
block = gr.Blocks(css=css)

options = ['SD2.1']

with block:
    gr.Markdown("# SD 2.1 vs. FreeU")
    with gr.Group():
        with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
            with gr.Column():
                text = gr.Textbox(
                    label="Enter your prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
            btn = gr.Button("Generate image", scale=0)
        sd_options = gr.Dropdown(options, label="SD options", value="SD2.1", visible=False)

    with gr.Group():
        with gr.Row():
            with gr.Accordion('FreeU Parameters (adjust based on prompt): ', open=False):
                with gr.Row():
                    b1 = gr.Slider(label='b1: Decoder stage 1 backbone factor',
                                   minimum=1, maximum=1.6, step=0.01, value=1.1)
                    b2 = gr.Slider(label='b2: Decoder stage 2 backbone factor',
                                   minimum=1, maximum=1.6, step=0.01, value=1.2)
                with gr.Row():
                    s1 = gr.Slider(label='s1: Decoder stage 1 skip factor',
                                   minimum=0, maximum=1, step=0.1, value=0.2)
                    s2 = gr.Slider(label='s2: Decoder stage 2 skip factor',
                                   minimum=0, maximum=1, step=0.1, value=0.2)
                seed = gr.Slider(label='Seed', minimum=0, maximum=1000, step=1, value=42)

    # Display the generated images
    with gr.Row():
        with gr.Group():
            with gr.Row():
                with gr.Column() as c1:
                    image_1 = gr.Image(interactive=False)
                    image_1_label = gr.Markdown("SD Image")
        with gr.Group():
            with gr.Row():
                with gr.Column() as c2:
                    image_2 = gr.Image(interactive=False)
                    image_2_label = gr.Markdown("FreeU Image")

    # Add examples and link them to the function
    ex = gr.Examples(examples=examples, fn=infer, inputs=[text, sd_options, seed, b1, b2, s1, s2], outputs=[image_1, image_2], cache_examples=False)
    ex.dataset.headers = [""]

    # Event listeners for submission and button click
    text.submit(infer, inputs=[text, sd_options, seed, b1, b2, s1, s2], outputs=[image_1, image_2])
    btn.click(infer, inputs=[text, sd_options, seed, b1, b2, s1, s2], outputs=[image_1, image_2])

block.launch()
