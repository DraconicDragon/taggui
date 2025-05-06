from pathlib import Path

from PySide6.QtCore import QModelIndex, Signal

from auto_captioning.auto_captioning_model import AutoCaptioningModel
from auto_captioning.models_list import get_model_class
from auto_captioning.models.wd_tagger import WdTagger
from models.image_list_model import ImageListModel
from utils.enums import CaptionPosition
from utils.image import Image
from utils.settings import get_tag_separator
from utils.ModelThread import ModelThread


def add_caption_to_tags(tags: list[str], caption: str,
                        caption_position: CaptionPosition) -> list[str]:
    if caption_position == CaptionPosition.DO_NOT_ADD or not caption:
        return tags
    tag_separator = get_tag_separator()
    new_tags = caption.split(tag_separator)
    # Make a copy of the tags so that the tags in the image list model are not
    # modified.
    tags = tags.copy()
    if caption_position == CaptionPosition.BEFORE_FIRST_TAG:
        tags[:0] = new_tags
    elif caption_position == CaptionPosition.AFTER_LAST_TAG:
        tags.extend(new_tags)
    elif caption_position == CaptionPosition.OVERWRITE_FIRST_TAG:
        if tags:
            tags[:1] = new_tags
        else:
            tags = new_tags
    elif caption_position == CaptionPosition.OVERWRITE_ALL_TAGS:
        tags = new_tags
    return tags


class CaptioningThread(ModelThread):
    # The image index, the caption, and the tags with the caption added. The
    # third parameter must be declared as `list` instead of `list[str]` for it
    # to work.
    caption_generated = Signal(QModelIndex, str, list)
    show_tags_popup = Signal(str)

    def __init__(self, parent, image_list_model: ImageListModel,
                 selected_image_indices: list[QModelIndex],
                 caption_settings: dict, tag_separator: str,
                 models_directory_path: Path | None):
        super().__init__(parent, image_list_model, selected_image_indices)
        self.caption_settings = caption_settings
        self.tag_separator = tag_separator
        self.models_directory_path = models_directory_path
        self.model: AutoCaptioningModel | None = None
        self.wd_tagger_model: WdTagger | None = None
        self.is_furrence_model = "furrence" in caption_settings['model_id'].lower()
        self.use_wd_tagger = (self.is_furrence_model and 
                            caption_settings.get('furrence_settings', {}).get('use_wd_tagger', False))

    def load_model(self):
        # Load WD Tagger first if needed
        if self.is_furrence_model and self.use_wd_tagger:
            wd_tagger_settings = {
                'model_id': self.caption_settings['furrence_settings']['wd_tagger_model'],
                'device': self.caption_settings['device'],
                'gpu_index': self.caption_settings['gpu_index'],
                'prompt': '',  # Empty prompt for compatibility
                'skip_hash': False,
                'caption_start': '',
                'caption_position': CaptionPosition.DO_NOT_ADD,
                'load_in_4_bit': self.caption_settings.get('load_in_4_bit', False),
                'limit_to_crop': False,
                'remove_tag_separators': False,
                'bad_words': '',
                'forced_words': '',
                'generation_parameters': {
                    'num_beams': 1,  # Default value
                    'min_new_tokens': 1,
                    'max_new_tokens': 100,
                    'length_penalty': 1.0,
                    'do_sample': False,
                    'temperature': 1.0,
                    'top_k': 50,
                    'top_p': 1.0,
                    'repetition_penalty': 1.0,
                    'no_repeat_ngram_size': 0
                },
                'wd_tagger_settings': {
                    'show_probabilities': self.caption_settings['furrence_settings']['wd_tagger_settings']['show_probabilities'],
                    'min_probability': self.caption_settings['furrence_settings']['wd_tagger_settings']['min_probability'],
                    'max_tags': self.caption_settings['furrence_settings']['wd_tagger_settings']['max_tags'],
                    'tags_to_exclude': self.caption_settings['furrence_settings']['wd_tagger_settings']['tags_to_exclude']
                }
            }
            self.wd_tagger_model = WdTagger(
                captioning_thread_=self,
                caption_settings=wd_tagger_settings
            )
            self.wd_tagger_model.load_processor_and_model()
            self.write("Loaded WD Tagger model for grounding")

        # Load main model
        model_id = self.caption_settings['model_id']
        model_class = get_model_class(model_id)
        self.model = model_class(
            captioning_thread_=self, caption_settings=self.caption_settings)
        self.error_message = self.model.get_error_message()
        if self.error_message:
            self.is_error = True
            return
        self.model.load_processor_and_model()
        self.model.monkey_patch_after_loading()
        self.device = self.model.device
        self.text = {
            'Generating': self.model.get_generation_text(),
            'generating': 'captioning'
        }

    def process_image(self, image_index, image: Image):
        try:
            # Step 1: Run WD Tagger if enabled
            grounding_tags = None
            if self.is_furrence_model and self.use_wd_tagger and self.wd_tagger_model:
                _, wd_tagger_inputs = self.wd_tagger_model.get_model_inputs(image)
                grounding_tags, _ = self.wd_tagger_model.generate_caption(wd_tagger_inputs, None)
                
                # Pass tags to Furrence2
                if hasattr(self.model, 'set_grounding_tags'):
                    self.model.set_grounding_tags(grounding_tags)
                
                # Show popup with tags
                self.show_tags_popup.emit(grounding_tags)  # Add this signal to the class

            # Step 2: Prepare inputs for main model
            image_prompt, model_inputs = self.get_model_inputs(image)
            
            # Step 3: Add grounding tags to inputs if available
            if grounding_tags and hasattr(self.model, 'set_grounding_tags'):
                self.model.set_grounding_tags(grounding_tags)
            
            # Step 4: Generate caption with main model
            self.write(f"Generating caption for image {image_index.row()+1}...")
            caption, console_output = self.generate_output(image_index, image, image_prompt, model_inputs)
            
            return True
        except Exception as e:
            self.error_message = str(e)
            self.is_error = True
            return False

    def get_model_inputs(self, image: Image):
        image_prompt = self.model.get_image_prompt(image)
        crop = self.caption_settings['limit_to_crop']
        return image_prompt, self.model.get_model_inputs(image_prompt, image, crop)

    def generate_output(self, image_index, image: Image, image_prompt: str | None, model_inputs) -> str:
        caption_position = self.caption_settings['caption_position']
        caption, console_output_caption = self.model.generate_caption(model_inputs, image_prompt)
        tags = add_caption_to_tags(image.tags, caption, caption_position)
        self.caption_generated.emit(image_index, caption, tags)
        return console_output_caption
