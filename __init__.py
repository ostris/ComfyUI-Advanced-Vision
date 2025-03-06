import os
import logging
import comfy.clip_vision
from comfy.utils import load_torch_file
import folder_paths


def load_advanced_vision_from_sd(sd, prefix="", convert_keys=False):
    config_root = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "configs")
    json_config = None
    if convert_keys:
        sd = comfy.clip_vision.convert_to_transformers(sd, prefix)
    if "vision_model.encoder.layers.22.layer_norm1.weight" in sd:
        if sd["vision_model.encoder.layers.0.layer_norm1.weight"].shape[0] == 1152:
            if sd["vision_model.embeddings.position_embedding.weight"].shape[0] == 1024:
                json_config = os.path.join(
                    config_root,
                    "clip_vision_siglip2_so400m_512.json"
                )
                print("Advanced Vision Model: clip_vision_siglip2_so400m_512 detected")

    if json_config is None:
        return None

    clip = comfy.clip_vision.ClipVisionModel(json_config)
    m, u = clip.load_sd(sd)
    if len(m) > 0:
        logging.warning("missing clip vision: {}".format(m))
    u = set(u)
    keys = list(sd.keys())
    for k in keys:
        if k not in u:
            sd.pop(k)
    return clip


def load_advanced_vision(ckpt_path):
    sd = load_torch_file(ckpt_path)
    if "visual.transformer.resblocks.0.attn.in_proj_weight" in sd:
        return load_advanced_vision_from_sd(sd, prefix="visual.", convert_keys=True)
    else:
        return load_advanced_vision_from_sd(sd)


class AdvancedVisionLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("clip_vision"), ),
            }}
    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "load_vision"

    CATEGORY = "loaders"

    def load_vision(self, clip_name):
        clip_path = folder_paths.get_full_path_or_raise(
            "clip_vision",
            clip_name
        )
        # try to load it through ours first
        clip_vision = load_advanced_vision(clip_path)

        # load it through comfy
        if clip_vision is None:
            clip_vision = comfy.clip_vision.load(clip_path)
        return (clip_vision,)


NODE_CLASS_MAPPINGS = {
    "AdvancedVisionLoader": AdvancedVisionLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedVisionLoader": "Load Advanced Vision Model",
}
