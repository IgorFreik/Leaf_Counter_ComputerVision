import torch.nn.functional as F
import gradio as gr
from torchvision import transforms
import torch
import numpy as np


PAD_BASE = 16


# Create padding functions to make input images compatible with models
def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w

    lh, uh = int((new_h - h) / 2), (new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), (new_w - w) - int((new_w - w) / 2)
    pads = (lw, uw, lh, uh)

    out = F.pad(x, pads, "constant", 0)

    return out, pads


def un_pad(x, pad):
    if pad[2] + pad[3] > 0:
        x = x[:, :, pad[2]: -pad[3], :]
    if pad[0] + pad[1] > 0:
        x = x[:, :, :, pad[0]: -pad[1]]
    return x


def predict_leaf_number(model, im):
    """
    Predicts the number of leaves on an image.

    :param img: png image to predict leaf number.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img = np.moveaxis(im, [2], [0]) / 255
    preprocessing = transforms.Compose([torch.Tensor,
                                        transforms.Resize((480, 480))])

    img = preprocessing(img).unsqueeze(0).to(device)
    mask = model.segmenter(img)
    leaf_num = int(model.counter(torch.cat((img, mask), axis=1)).item())
    return leaf_num


def show_web_interface(model):
    """
    Runs a gradio web interface for the model.

    :param model: a LeafCounter class instance.
    """
    with gr.Blocks() as demo:
        gr.Markdown("# Leaf counter with UNet++ and ResNet50")

        with gr.Row():
            im = gr.Image()
            txt = gr.Textbox(value="", label="Output")

        predict_fn = lambda img: predict_leaf_number(model, img)

        btn = gr.Button(value="Count leafs")
        btn.click(predict_fn, inputs=[im], outputs=txt)

        gr.Markdown("## Image Examples")

        gr.Examples(
            examples=["../examples/leaf_im_big.png"],
            inputs=im,
            outputs=txt,
            fn=predict_fn
        )
    demo.launch()
