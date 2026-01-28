import torch
import torchvision.transforms as T

from PIL import Image
from torchvision.transforms import InterpolationMode
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ============================================================================
# Image preprocessing utilities (for InternVL3.5)
# ============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ============================================================================
# Query utilities
# ============================================================================

def get_query(category):
    if category == 1:
        query = "Strictly answer with a single word only: Is there a pedestrian ahead? Possible answers: [Yes, No]"
    elif category == 2:
        query = "Strictly answer with a single word only: In which direction is the pedestrian walking? Possible answers: [Left, Right]"
    elif category == 3:
        query = "Strictly answer with a single word only: How many pedestrians are ahead? Possible answers: [Zero, One, Two, Three, Four]"
    elif category == 4:
        query = "Strictly answer with a single word only: Which of the truck's blinkers is on? Possible answers: [Left, Right]"
    elif category == 5:
        query = "Strictly answer with a single word only: On which side of the road is the pedestrian walking? Possible answers: [Left, Right]"
    elif category == 6:
        query = "Strictly answer with a single word only: Is there a traffic barrel ahead? Possible answers: [Yes, No]"
    elif category == 7:
        query = "Strictly answer with a single word only: How many traffic barrels are ahead? Possible answers: [Zero, One, Two, Three, Four]"
    elif category == 8:
        query = "Strictly answer with a single word only: In which direction is the bicycle moving? Possible answers: [Left, Right]"
    return query

# ============================================================================
# Model loading functions
# ============================================================================

def load_model(model_name, args):
    """Load model and processor/tokenizer based on model name."""
    if model_name == "ovis2.5":
        return _load_ovis2_5()
    elif model_name == "internvl3.5":
        return _load_internvl3_5()
    elif model_name == "vst":
        return _load_vst(args.version)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def _load_ovis2_5():
    path = "AIDC-AI/Ovis2.5-2B"
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda()
    return model, None  # Ovis uses model's built-in preprocessor

def _load_internvl3_5():
    path = 'OpenGVLab/InternVL3_5-2B'
    model = AutoModel.from_pretrained(
        path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    return model, tokenizer

def _load_vst(version):
    if version == "rl":
        path = "rayruiyang/VST-3B-RL"
    elif version == "sft":
        path = "rayruiyang/VST-3B-SFT"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    processor = AutoProcessor.from_pretrained(path, min_pixels=256*28*28, max_pixels=1280*28*28)
    return model, processor

# ============================================================================
# Input preprocessing functions
# ============================================================================

def preprocess_inputs(model_name, model, processor, annotation, question, args):
    """Preprocess inputs based on model type."""
    if model_name == "ovis2.5":
        return _preprocess_ovis2_5(model, annotation, question)
    elif model_name == "internvl3.5":
        return _preprocess_internvl3_5(model, processor, annotation, question, args.num_tiles)
    elif model_name == "vst":
        return _preprocess_vst(processor, annotation, question)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def _preprocess_ovis2_5(model, annotation, question):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": Image.open(annotation["image_path"])},
            {"type": "text", "text": question},
        ],
    }]

    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=False
    )
    input_ids = input_ids.cuda()
    pixel_values = pixel_values.cuda() if pixel_values is not None else None
    grid_thws = grid_thws.cuda() if grid_thws is not None else None
    attention_mask = torch.ne(input_ids, model.text_tokenizer.pad_token_id).to(device=input_ids.device)
    
    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "grid_thws": grid_thws,
        "attention_mask": attention_mask
    }

def _preprocess_internvl3_5(model, tokenizer, annotation, question, num_tiles):
    template = "<|im_start|>system\n你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。<|im_end|>\n<|im_start|>user\n<img><IMG_CONTEXT></img>\n<question><|im_end|>\n<|im_start|>assistant\n"
    img_context_tokens_num = num_tiles * 256
    query = template.replace("<question>", question)
    img_context = "<IMG_CONTEXT>" * img_context_tokens_num
    query = query.replace("<IMG_CONTEXT>", img_context)
    
    inputs = tokenizer(query, return_tensors='pt')
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()
    image_flags = torch.ones(num_tiles, 256, 2048).cuda()
    
    img = load_image(annotation["image_path"]).to(torch.bfloat16).cuda()
    
    return {
        "pixel_values": img,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_flags": image_flags
    }

def _preprocess_vst(processor, annotation, question):
    from qwen_vl_utils import process_vision_info
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": annotation["image_path"],
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    return inputs

# ============================================================================
# Hook registration functions
# ============================================================================

def register_hooks(model_name, model, save_hook, initial_inputs):
    """Register forward hooks for feature extraction."""
    if model_name == "ovis2.5":
        _register_hooks_ovis2_5(model, save_hook, initial_inputs)
    elif model_name == "internvl3.5":
        _register_hooks_internvl3_5(model, save_hook, initial_inputs)
    elif model_name == "vst":
        _register_hooks_vst(model, save_hook, initial_inputs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def _register_hooks_ovis2_5(model, save_hook, inputs):
    # Vision encoder hooks
    vision_encoder_layers = list(range(len(model.visual_tokenizer.vit.vision_model.encoder.layers)))
    for layer in vision_encoder_layers:
        model.visual_tokenizer.vit.vision_model.encoder.layers[layer].register_forward_hook(
            save_hook(f"vision_encoder/layer_{layer + 1}", "vision_encoder")
        )
    
    model.visual_tokenizer.vit.vision_model.register_forward_hook(
        save_hook(f"vision_encoder/post_layer_norm", "vision_encoder")
    )
    
    # Projector hook
    model.vte.register_forward_hook(save_hook("projector", "projector"))
    
    # Language model hooks
    indices = (inputs["input_ids"] == -300).nonzero(as_tuple=True)[1]
    language_model_layers = list(range(len(model.llm.model.layers)))
    for layer in language_model_layers:
        model.llm.model.layers[layer].register_forward_hook(
            save_hook(f"language_model/layer_{layer + 1}", "llm", indices=indices)
        )
    
    model.llm.model.register_forward_hook(
        save_hook(f"language_model/post_layer_norm", "llm", indices=indices)
    )

def _register_hooks_internvl3_5(model, save_hook, inputs):
    # Vision encoder hooks
    vision_encoder_layers = list(range(len(model.vision_model.encoder.layers)))
    for layer in vision_encoder_layers:
        model.vision_model.encoder.layers[layer].register_forward_hook(
            save_hook(f"vision_encoder/layer_{layer + 1}", "vision_encoder")
        )
    
    # Projector hook
    model.mlp1.register_forward_hook(save_hook("projector", "projector"))
    
    # Language model hooks
    indices = (inputs["input_ids"] == model.img_context_token_id).nonzero(as_tuple=True)[1]
    language_model_layers = list(range(len(model.language_model.model.layers)))
    for layer in language_model_layers:
        model.language_model.model.layers[layer].register_forward_hook(
            save_hook(f"language_model/layer_{layer + 1}", "llm", indices=indices)
        )
    
    model.language_model.model.register_forward_hook(
        save_hook(f"language_model/post_layer_norm", "llm", indices=indices)
    )

def _register_hooks_vst(model, save_hook, inputs):
    # Vision encoder hooks
    vision_encoder_layers = list(range(len(model.model.visual.blocks)))
    for layer in vision_encoder_layers:
        model.model.visual.blocks[layer].register_forward_hook(
            save_hook(f"vision_encoder/layer_{layer + 1}", "vision_encoder")
        )
    
    # Projector hook
    model.model.visual.merger.register_forward_hook(save_hook("projector", "projector"))
    
    # Language model hooks
    indices = (inputs.input_ids[0] == 151655).nonzero(as_tuple=True)[0]
    language_model_layers = list(range(len(model.model.language_model.layers)))
    for layer in language_model_layers:
        model.model.language_model.layers[layer].register_forward_hook(
            save_hook(f"language_model/layer_{layer + 1}", "llm", indices=indices)
        )
    
    model.model.language_model.register_forward_hook(
        save_hook(f"language_model/post_layer_norm", "llm", indices=indices)
    )

# ============================================================================
# Inference functions
# ============================================================================

def run_inference(model_name, model, inputs):
    """Run model inference."""
    if model_name == "ovis2.5":
        return model.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            grid_thws=inputs["grid_thws"],
        )
    elif model_name == "internvl3.5":
        return model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_flags=inputs["image_flags"],
            output_hidden_states=False
        )
    elif model_name == "vst":
        return model(**inputs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ============================================================================
# Save hook functions
# ============================================================================

def get_save_hook(model_name, args, average_features):
    """Get the appropriate save hook for the model."""
    if model_name == "ovis2.5":
        return get_save_hook_ovis2_5(args.llm_features, average_features)
    elif model_name == "internvl3.5":
        return get_save_hook_internvl3_5(args.use_cls, args.llm_features, average_features)
    elif model_name == "vst":
        return get_save_hook_vst(args.llm_features, average_features)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_save_hook_internvl3_5(use_cls, llm_features, average_features):
    def save_hook(name, model_component, indices=None):
        def hook(module, inp, out):
            if model_component == "vision_encoder":
                if not use_cls:
                    average_features[name].append(out[:, 1:].mean(dim=(0, 1)).detach())
                else:
                    average_features[name].append(out.mean(dim=(0, 1)).detach())
            elif model_component == "projector":
                average_features[name].append(out.mean(dim=(0, 1)).detach())
            elif model_component == "llm":
                if "post_layer_norm" in name:
                    out = out.last_hidden_state
                elif isinstance(out, tuple):
                    out = out[0]
                if llm_features == "visual_embs":
                    average_features[name].append(out[:, indices].mean(dim=(0, 1)).detach())
                elif llm_features == "last_token":
                    average_features[name].append(out[0, -1].clone().detach())
                elif llm_features == "all":
                    average_features[name].append(out.mean(dim=(0, 1)).detach())
                elif llm_features == "visual_embs_and_last_token":
                    f1 = out[:, indices].mean(dim=(0, 1)).detach()
                    f2 = out[0, -1].clone().detach()
                    average_features[name].append(torch.cat([f1, f2], dim=0))
        return hook

    return save_hook

def get_save_hook_ovis2_5(llm_features, average_features):
    def save_hook(name, model_component, indices=None):
        def hook(module, inp, out):
            if model_component == "vision_encoder":
                if "post_layer_norm" in name:
                    out = out.last_hidden_state
                average_features[name].append(out.mean(dim=0).detach())
            elif model_component == "projector":
                if out.shape[0] > 4:
                    average_features[name].append(out.mean(dim=0).detach())
            elif model_component == "llm":
                if "post_layer_norm" in name:
                    out = out.last_hidden_state
                elif isinstance(out, tuple):
                    out = out[0]
                if llm_features == "visual_embs":
                    average_features[name].append(out[:, indices].mean(dim=(0, 1)).detach())
                elif llm_features == "last_token":
                    average_features[name].append(out[0, -1].clone().detach())
                elif llm_features == "all":
                    average_features[name].append(out.mean(dim=(0, 1)).detach())
                elif llm_features == "visual_embs_and_last_token":
                    f1 = out[:, indices].mean(dim=(0, 1)).detach()
                    f2 = out[0, -1].clone().detach()
                    average_features[name].append(torch.cat([f1, f2], dim=0))
        return hook

    return save_hook

def get_save_hook_vst(llm_features, average_features):
    def save_hook(name, model_component, indices=None):
        def hook(module, inp, out):
            if model_component == "vision_encoder":
                average_features[name].append(out.mean(dim=0).detach())
            elif model_component == "projector":
                average_features[name].append(out.mean(dim=0).detach())
            elif model_component == "llm":
                if "post_layer_norm" in name:
                    out = out.last_hidden_state
                elif isinstance(out, tuple):
                    out = out[0]
                if llm_features == "visual_embs":
                    average_features[name].append(out[:, indices].mean(dim=(0, 1)).detach())
                elif llm_features == "last_token":
                    average_features[name].append(out[0, -1].clone().detach())
                elif llm_features == "all":
                    average_features[name].append(out.mean(dim=(0, 1)).detach())
                elif llm_features == "visual_embs_and_last_token":
                    f1 = out[:, indices].mean(dim=(0, 1)).detach()
                    f2 = out[0, -1].clone().detach()
                    average_features[name].append(torch.cat([f1, f2], dim=0))
        return hook

    return save_hook

# ============================================================================
# Save path functions
# ============================================================================

def get_save_path(model_name, args):
    """Get the save path for features based on model and arguments."""
    if model_name == "ovis2.5":
        return f"{args.save_path}/ovis2.5/llm_{args.llm_features}"
    elif model_name == "internvl3.5":
        return f"{args.save_path}/internvl3.5/cls_{args.use_cls}/llm_{args.llm_features}"
    elif model_name == "vst":
        return f"{args.save_path}/vst_{args.version}/llm_{args.llm_features}"
    else:
        raise ValueError(f"Unknown model: {model_name}")
