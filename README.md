# ComfyUI Advanced Vision

This is a custom node for the [ComfyUI](https://github.com/comfyanonymous/ComfyUI) project to support loading more vision models.
It will fallback to the default loading if comfy supported models are detected. Meaning this node can be used as a drop-in replacement for the "Load Clip Vision" node.

The supported vision models can be found here at huggingface [ostris/ComfyUI-Advanced-Vision](https://huggingface.co/ostris/ComfyUI-Advanced-Vision). Put them in the `models/clip_vision` directory.

### Supported Models

- [google/siglip2-so400m-patch16-512](https://huggingface.co/google/siglip2-so400m-patch16-512)

### Installation

Clone this repo into the `custom_nodes` directory of the ComfyUI project.
