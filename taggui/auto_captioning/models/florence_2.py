import numpy as np
from auto_captioning.auto_captioning_model import AutoCaptioningModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, BatchFeature
from utils.image import Image
from utils.utils import list_with_and

class Florence2(AutoCaptioningModel):
    use_safetensors = None
    transformers_model_class = AutoModelForCausalLM
    task_prompts = [
        '<CAPTION>',
        '<DETAILED_CAPTION>',
        '<MORE_DETAILED_CAPTION>',
        '<OCR>'
    ]
    default_prompt = task_prompts[2]

    def get_additional_error_message(self) -> str | None:
        if self.prompt and self.prompt not in self.task_prompts:
            quoted_task_prompts = [f'"{task_prompt}"'
                                   for task_prompt in self.task_prompts]
            return (f'This model only supports the following prompts: '
                    f'{list_with_and(quoted_task_prompts)}. The default '
                    f'prompt is "{self.default_prompt}".')
        if self.caption_start:
            return 'This model does not support `Start caption with`.'
        return None

    def get_default_prompt(self) -> str:
        return self.default_prompt


class Florence2Promptgen(Florence2):
    use_safetensors = True
    task_prompts = [
        '<GENERATE_PROMPT>',
        '<CAPTION>',
        '<DETAILED_CAPTION>',
        '<MORE_DETAILED_CAPTION>'
    ]
    default_prompt = task_prompts[0]

class Furrence2(Florence2):
    use_safetensors = True
    task_prompts = [
        '<GENERATE_PROMPT>',
        '<CAPTION>',
        '<DETAILED_CAPTION>',
        '<MORE_DETAILED_CAPTION>'
    ]
    default_prompt = task_prompts[0]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grounding_tags = None
    
    def set_grounding_tags(self, tags: str):
        """Set the grounding tags from WD Tagger"""
        self.grounding_tags = tags
    
    def get_model_load_arguments(self) -> dict:
        arguments = {
            'trust_remote_code': True,
            'use_safetensors': self.use_safetensors,
            'torch_dtype': self.dtype
        }
        if self.load_in_4_bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True
            )
            arguments['quantization_config'] = quantization_config
        elif self.device.type == 'cuda':
            arguments['torch_dtype'] = self.dtype
        return arguments
    
    def load_model(self, model_load_arguments: dict):
        with self.model_load_context_manager:
            model = self.transformers_model_class.from_pretrained(
                self.model_id, **model_load_arguments)
        model.to(self.device)  # Ensure model is on the correct device
        model.eval()
        return model

    def get_model_inputs(self, image_prompt: str, image: Image,
                        crop: bool) -> BatchFeature | dict | np.ndarray:
        text = self.get_input_text(image_prompt)
        pil_image = self.load_image(image, crop)
        raw_inputs = self.processor(
            text=text, images=pil_image, return_tensors='pt'
        ).to(self.device)  # Move all inputs to the device at once
        
        model_inputs = {}
        for name, tensor in raw_inputs.items():
            # Cast the image pixels to float16 if needed
            if name == "pixel_values":
                tensor = tensor.to(dtype=self.dtype)
            model_inputs[name] = tensor
        
        # Add grounding tags if available
        if self.grounding_tags:
            model_inputs['grounding_tags'] = self.grounding_tags
        
        return model_inputs
    
    def generate_caption(self, model_inputs: BatchFeature | dict | np.ndarray,
                         image_prompt: str) -> tuple[str, str]:
        # Pre-process with grounding tags if available
        if hasattr(self, 'grounding_tags') and self.grounding_tags:
            # Add any Furrence-specific grounding processing here
            pass
            
        # Then call the parent's generate_caption
        return super().generate_caption(model_inputs, image_prompt)