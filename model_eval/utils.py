import torch
import torchvision.transforms as T
import warnings

from PIL import Image
from torchvision.transforms import InterpolationMode
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Model paths
MODEL_PATHS = {
    "ovis2.5": "AIDC-AI/Ovis2.5-2B",
    "internvl3.5": "OpenGVLab/InternVL3_5-2B",
    "vst-rl": "rayruiyang/VST-3B-RL",
    "vst-sft": "rayruiyang/VST-3B-SFT",
}


def load_model(model_name):
    """Load a model by name."""
    path = MODEL_PATHS[model_name]
    
    if model_name == "ovis2.5":
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).cuda()
    elif model_name == "internvl3.5":
        model = AutoModel.from_pretrained(
            path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval().cuda()
    elif model_name in ["vst-rl", "vst-sft"]:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).cuda()
    
    return model


def get_tokenizer(model_name, model=None):
    """Get the tokenizer for a model."""
    path = MODEL_PATHS[model_name]
    
    if model_name == "ovis2.5":
        return model.text_tokenizer
    elif model_name == "internvl3.5":
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        return tokenizer
    elif model_name in ["vst-rl", "vst-sft"]:
        processor = AutoProcessor.from_pretrained(path, min_pixels=256*28*28, max_pixels=1280*28*28)
        return processor
    

def get_answer_ids(tokenizer, answers):
    """Get token IDs for answer options."""
    # For VST models, tokenizer is actually the processor
    if hasattr(tokenizer, 'tokenizer'):
        return [tokenizer.tokenizer.encode(ans, add_special_tokens=False) for ans in answers]
    return [tokenizer.encode(ans, add_special_tokens=False) for ans in answers]


def prepare_inputs(model_name, model, tokenizer, question, image_path=None, static_only=False, static_inputs=None):
    """Prepare inputs for a model."""
    
    if model_name == "ovis2.5":
        return _prepare_ovis_inputs(model, question, image_path, static_only)
    elif model_name == "internvl3.5":
        return _prepare_internvl_inputs(tokenizer, question, image_path, static_only, static_inputs)
    elif model_name in ["vst-rl", "vst-sft"]:
        return _prepare_vst_inputs(tokenizer, question, image_path, static_only)


def _prepare_ovis_inputs(model, question, image_path, static_only):
    """Prepare inputs for Ovis model."""
    if static_only:
        return None
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": Image.open(image_path)},
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
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "grid_thws": grid_thws,
    }


def _prepare_internvl_inputs(tokenizer, question, image_path, static_only, static_inputs):
    """Prepare inputs for InternVL model."""
    if static_only:
        # Prepare static inputs (tokenized query template)
        template = "<|im_start|>system\n你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。<|im_end|>\n<|im_start|>user\n<img><IMG_CONTEXT></img>\n<question><|im_end|>\n<|im_start|>assistant\n"
        img_context_tokens_num = 9 * 256
        query = template.replace("<question>", question)
        img_context = "<IMG_CONTEXT>" * img_context_tokens_num
        query = query.replace("<IMG_CONTEXT>", img_context)
        inputs = tokenizer(query, return_tensors='pt')
        input_ids = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()
        image_flags = torch.ones(9, 256, 2048).cuda()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_flags": image_flags,
        }
    
    # Load image and combine with static inputs
    image = load_image(image_path).to(torch.bfloat16).cuda()
    
    return {
        "input_ids": static_inputs["input_ids"],
        "attention_mask": static_inputs["attention_mask"],
        "image_flags": static_inputs["image_flags"],
        "pixel_values": image,
    }


def _prepare_vst_inputs(processor, question, image_path, static_only):
    """Prepare inputs for VST model."""
    if static_only:
        return None
    
    from qwen_vl_utils import process_vision_info
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
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


def forward_pass(model_name, model, inputs):
    """Perform a forward pass through the model."""
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
    elif model_name in ["vst-rl", "vst-sft"]:
        return model(**inputs)


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


def get_query(category, question_only=False):
    if not question_only:
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
    else:
        if category == 1:
            query = "Is there a pedestrian ahead?"
        elif category == 2:
            query = "In which direction is the pedestrian walking?"
        elif category == 3:
            query = "How many pedestrians are ahead?"
        elif category == 4:
            query = "Which of the truck's blinkers is on?"
        elif category == 5:
            query = "On which side of the road is the pedestrian walking?"
        elif category == 6:
            query = "Is there a traffic barrel ahead?"
        elif category == 7:
            query = "How many traffic barrels are ahead?"
        elif category == 8:
            query = "In which direction is the bicycle moving?"
        return query

def get_answer(category, label):
    if category in [1, 6]:
        answers = {0: "No", 1: "Yes"}
    elif category in [2, 4, 5, 8]:
        answers = {0: "Left", 1: "Right"}
    elif category in [3, 7]:
        answers = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four"}

    return answers[label]
