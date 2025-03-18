# IF_VideoPromptsNode.py
import os
import sys
import json
import torch
import logging
import hashlib
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from PIL import Image, ImageOps
import numpy as np
import folder_paths
import gc
import qwen_vl_utils
try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoConfig
    QWEN_AVAILABLE = True
except ImportError:
    logging.warning("Transformers package not found. Please install with: pip install transformers torch")
    QWEN_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoPromptNode:
    """
    A ComfyUI node that analyzes video sequences or video files using Qwen2.5-VL multimodal models.
    
    This node provides two main modes of operation:
    1. Frame-based mode: Takes pre-loaded frames from LoadVideo nodes and analyzes them
    2. Direct video mode: Takes a video file directly and processes it
    
    The node can generate:
    - Descriptive prompts for video content
    - Complete scene analysis
    - Key scene breakdowns
    - Negative prompts based on configured templates
    """
    def __init__(self):
        # Initialize paths for storing presets and profiles
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.presets_dir = os.path.join(self.current_dir, "presets")
        self.profiles_path = os.path.join(self.presets_dir, "profiles.json")
        
        # Create directory if it doesn't exist
        os.makedirs(self.presets_dir, exist_ok=True)
        
        # Load profiles
        self.profiles = self.load_presets(self.profiles_path)
        
        # Check if LLM folder exists in models, create if not
        llm_path = os.path.join(folder_paths.models_dir, "LLM")
        os.makedirs(llm_path, exist_ok=True)
        
        # Add LLM path to folder_paths if not already there
        if hasattr(folder_paths, "folder_names_and_paths"):
            if "LLM" not in folder_paths.folder_names_and_paths:
                # Register LLM path to folder_paths
                supported_extensions = {'.pt', '.pth', '.safetensors', '.bin', '.ckpt'}
                folder_paths.folder_names_and_paths["LLM"] = ([llm_path], supported_extensions)
        
        # Default values
        self.model = None
        self.processor = None
        self.device = self.get_optimal_device()
        self.qwen_models = {
            "Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen2.5-VL-3B-Instruct-AWQ": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
            "Qwen2.5-VL-7B-Instruct-AWQ": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
        }
        
        # System prompts for different languages
        self.system_prompts = {
            "en": "You are a professional video sequence analyzer. Describe the visual content of the frames with attention to detail, capturing the storytelling, composition, lighting, movement, and emotional tone. Be specific, clear, and concise.",
            "zh": "你是一位专业的视频序列分析师。请详细描述帧中的视觉内容，注意讲故事、构图、光线、运动和情感基调。请具体、清晰、简洁。"
        }
        
        # Load negative prompts
        self.neg_prompts_path = os.path.join(self.presets_dir, "neg_prompts.json")
        self.neg_prompts = self.load_neg_prompts()

    def get_optimal_device(self):
        """Determine the best device for model loading based on system capabilities."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # For Mac M1/M2 chips
            return "mps"
        else:
            return "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        # Get available local models
        llm_path = os.path.join(folder_paths.models_dir, "LLM")
        local_models = []
        
        # Look for locally downloaded models in the LLM directory
        if os.path.exists(llm_path):
            for model_dir in os.listdir(llm_path):
                if os.path.isdir(os.path.join(llm_path, model_dir)):
                    # Check if it contains a config.json file (transformer model)
                    if os.path.exists(os.path.join(llm_path, model_dir, "config.json")):
                        local_models.append(f"local:{model_dir}")
        
        # Get available negative prompts
        instance = cls()
        neg_prompt_keys = ["None"] + list(instance.neg_prompts.keys())
        
        # Combine with remote models
        model_choices = [
            "Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-7B-Instruct",
            "Qwen2.5-VL-3B-Instruct-AWQ",
            "Qwen2.5-VL-7B-Instruct-AWQ"
        ] + local_models
        
        # Get input directories for video files
        input_dir = folder_paths.get_input_directory()
        video_files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1].lower() in ['mp4', 'avi', 'mov', 'webm', 'mkv']):
                    video_files.append(f)
        
        # Sort video files for easier selection
        video_files = sorted(video_files)
        
        return {
            "required": {
                "input_mode": (["Frames", "Video File"], {"default": "Frames", "tooltip": "Select input mode: use pre-loaded frames or direct video file"}),
                "model_name": (model_choices, {"default": "Qwen2.5-VL-3B-Instruct-AWQ", "tooltip": "Select the Qwen2.5-VL model to use"}),
                "profile": (["None"] + list(instance.profiles.keys()), {"default": "HyVideoAnalyzer - Simple one line prompt", "tooltip": "Select a profile with predefined system prompt and rules"}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "tooltip": "Maximum number of new tokens to generate"}),
                "frame_sample_count": ("INT", {"default": 16, "min": 1, "max": 32, "step": 1, "tooltip": "Number of frames to sample from entire sequence"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Higher values increase creativity but reduce coherence"}),
                "analysis_type": (["Full sequence", "Key scenes", "Single summary"], {"default": "Full sequence", "tooltip": "Type of analysis to perform on the video"}),
                "language": (["English", "Chinese"], {"default": "English", "tooltip": "Language for the output"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Input frames from a LoadVideo node"}),
                "video_file": (sorted(video_files), {"tooltip": "Select a video file from the input directory"}),
                "fps": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 60.0, "step": 0.1, "tooltip": "Frames per second for video processing. Higher values sample more frames."}),
                "max_pixels": ("INT", {"default": 512*512, "min": 0, "max": 1280*720, "step": 1000, "tooltip": "Max pixels for video processing (0 = default)"}),
                "fallback_frame_count": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1, "tooltip": "Number of frames to use in fallback mode if initial processing fails. Lower values use less VRAM."}),
                "custom_system_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Custom system prompt to override the profile"}),
                "prefix": ("STRING", {"default": "", "tooltip": "Text to add before the generated prompt"}),
                "suffix": ("STRING", {"default": "", "tooltip": "Text to add after the generated prompt"}),
                "seed": ("INT", {"default": -1, "tooltip": "Random seed for generation (use -1 for random)"}),
                "negative_prompt": (neg_prompt_keys, {"default": "None", "tooltip": "Predefined negative prompt to use"}),
                "model_offload": (["Yes", "No"], {"default": "Yes", "tooltip": "Offload model from GPU when not in use to save VRAM"}),
                "precision": (["float16", "bfloat16", "float32"], {"default": "float16", "tooltip": "Model precision - lower precision uses less VRAM but may reduce quality"})
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("sequence_description", "scene_breakdown", "preview_image", "negative_prompt")
    FUNCTION = "analyze_sequence"
    CATEGORY = "ImpactFrames💥🎞️/LLM"

    def load_model(self, model_name, precision="float16", model_offload="Yes"):
        """Load and prepare the model."""
        if self.model is not None:
            # Check if we need to reinitialize
            if getattr(self, "current_model_name", None) == model_name and \
               getattr(self, "current_precision", None) == precision:
                logger.info(f"Model {model_name} already loaded, skipping")
                
                # If we're using CUDA, make sure the model is on CUDA
                if self.device == "cuda" and hasattr(self.model, "device") and self.model.device.type != "cuda":
                    logger.info(f"Moving model from {self.model.device} to cuda:0")
                    try:
                        self.model = self.model.to("cuda:0")
                    except Exception as e:
                        logger.warning(f"Failed to move model to CUDA: {e}")
                        # We'll try again later in analyze_sequence
                
                return
            else:
                # Clean up old model before loading a new one
                logger.info(f"Unloading previous model {getattr(self, 'current_model_name', 'unknown')}")
                del self.model
                del self.processor
                torch.cuda.empty_cache()
                gc.collect()
                self.model = None
                self.processor = None
        
        try:
            logger.info(f"Loading model: {model_name} with precision {precision}")
            
            # Set appropriate tensor type based on precision setting
            if precision == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif precision == "float32":
                dtype = torch.float32
            else:
                dtype = torch.float16
                
            # Determine if we can use flash attention
            can_use_flash_attn = False
            if self.device == "cuda":
                try:
                    from flash_attn import flash_attn_func
                    can_use_flash_attn = True
                    logger.info("Flash attention available, will use for better performance")
                except ImportError:
                    pass
            
            # Handle local models
            if model_name.startswith("local:"):
                local_model_dir = os.path.join(folder_paths.models_dir, "LLM", model_name[6:])
                logger.info(f"Loading local model from: {local_model_dir}")
                
                # Check for a specific model_type.txt file that could indicate special loading requirements
                model_type_path = os.path.join(local_model_dir, "model_type.txt")
                if os.path.exists(model_type_path):
                    with open(model_type_path, "r") as f:
                        model_type = f.read().strip()
                else:
                    model_type = "default"
                
                # Setup model loading parameters
                model_kwargs = {
                    "torch_dtype": dtype,
                    "trust_remote_code": True,
                }
                
                # Configure device mapping based on hardware
                if self.device == "cuda":
                    # Always use a single GPU setup to avoid device conflicts
                    logger.info("Using single GPU configuration to avoid device conflicts")
                    # Don't use device_map at all - load to a specific device instead
                    model_kwargs.pop("device_map", None)
                else:
                    # For CPU or MPS, use the device directly
                    logger.info(f"Using {self.device} device")
                    model_kwargs["device_map"] = {"": self.device}
                
                # Add flash attention if available
                if can_use_flash_attn:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                
                # Load the model
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    local_model_dir,
                    **model_kwargs
                )
                self.processor = AutoProcessor.from_pretrained(
                    local_model_dir,
                    trust_remote_code=True
                )
                    
            # Handle remote Qwen models
            elif model_name in self.qwen_models:
                # Get the HF model name
                hf_model_name = self.qwen_models[model_name]
                
                # Setup model loading parameters
                model_kwargs = {
                    "torch_dtype": dtype,
                    "trust_remote_code": True,
                }
                
                # Configure device mapping based on hardware
                if self.device == "cuda":
                    # Always use a single GPU setup to avoid device conflicts
                    logger.info("Using single GPU configuration to avoid device conflicts")
                    # Don't use device_map at all - load to a specific device instead
                    model_kwargs.pop("device_map", None)
                else:
                    # For CPU or MPS, use the device directly
                    logger.info(f"Using {self.device} device")
                    model_kwargs["device_map"] = {"": self.device}
                
                # Add flash attention if available and appropriate
                if can_use_flash_attn:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                
                # Special handling for AWQ models
                if "AWQ" in model_name:
                    logger.info("Loading AWQ quantized model")
                    model_kwargs["torch_dtype"] = dtype  # AWQ models still need dtype
                    
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    hf_model_name,
                    **model_kwargs
                )
                
                # Load processor with appropriate settings
                # Use default pixel limits for processor
                min_pixels = 256 * 28 * 28  # Minimum pixel dimensions (default from Qwen docs)
                max_pixels = 1280 * 28 * 28  # Maximum pixel dimensions (default from Qwen docs)
                
                self.processor = AutoProcessor.from_pretrained(
                    hf_model_name,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    trust_remote_code=True
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")
                
            # Store the model name for future reference
            self.current_model_name = model_name
            self.current_precision = precision
            
            # For CUDA, explicitly move model to cuda:0 after loading
            if self.device == "cuda" and not hasattr(self.model, "device_map"):
                logger.info("Moving model to cuda:0")
                self.model = self.model.to("cuda:0")
            
            logger.info(f"Model loaded successfully with device_map: {self.model.device_map if hasattr(self.model, 'device_map') else self.device}")
                
        except Exception as e:
            import traceback
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load model: {e}")

    def ensure_model_downloaded(self, model_name):
        """
        Ensures the model is downloaded to the models/LLM directory.
        Returns the local path to the model.
        """
        if not model_name.startswith("local:"):
            model_id = self.qwen_models.get(model_name, model_name)
            local_dir = os.path.join(folder_paths.models_dir, "LLM", model_id.split('/')[-1])
            
            if not os.path.exists(local_dir):
                try:
                    from huggingface_hub import snapshot_download
                    
                    # Create directory
                    os.makedirs(local_dir, exist_ok=True)
                    
                    logger.info(f"Downloading {model_id} to {local_dir}...")
                    snapshot_download(
                        repo_id=model_id,
                        local_dir=local_dir,
                        local_dir_use_symlinks=False
                    )
                    logger.info(f"Model downloaded successfully to {local_dir}")
                except Exception as e:
                    logger.error(f"Error downloading model: {e}")
                    raise
            
            return local_dir
        else:
            # For local models, just return the path
            local_model_dir = model_name[6:]  # Remove "local:" prefix
            return os.path.join(folder_paths.models_dir, "LLM", local_model_dir)

    def load_presets(self, file_path: str) -> Dict[str, Any]:
        """
        Load JSON presets with support for multiple encodings and better error handling.
        """
        
        # Try to load existing file with different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'gbk']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    data = json.loads(content)
                    return data
            except Exception:
                continue
                
        logger.error(f"Error: Failed to load {file_path} with any supported encoding")
        return {}

    def get_system_prompt(self, profile: str, custom_prompt: str = None, language: str = "English") -> str:
        """Get the system prompt from a profile, custom prompt, or language-specific default."""
        # If custom prompt is provided, use it
        if custom_prompt and custom_prompt.strip():
            return custom_prompt.strip()
        
        # If profile is specified and exists, use it
        if profile != "None" and profile in self.profiles:
            profile_content = self.profiles.get(profile, {})
            if isinstance(profile_content, str):
                return profile_content
            elif isinstance(profile_content, dict):
                # Try to construct from instruction and rules
                instruction = profile_content.get("instruction", "")
                rules = profile_content.get("rules", [])
                
                if instruction and rules:
                    return instruction + "\n\n" + "\n".join([f"- {rule}" for rule in rules])
                elif instruction:
                    return instruction
        
        # Fallback to language-specific system prompt
        lang_key = "zh" if language == "Chinese" else "en"
        return self.system_prompts.get(lang_key, self.system_prompts["en"])

    def load_neg_prompts(self) -> Dict[str, str]:
        """Load negative prompts from JSON file or create defaults if not exists."""
        
        # Try to load existing file with different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'gbk']
        
        for encoding in encodings:
            try:
                with open(self.neg_prompts_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    data = json.loads(content)
                    return data
            except Exception:
                continue
                
        logger.error(f"Error: Failed to load {self.neg_prompts_path} with any supported encoding")
        return {}

    def process_frames(self, images_tensor, frame_sample_count, max_pixels=512*512):
        """
        Process image frames for the model by sampling and resizing.
        
        Args:
            images_tensor: Input tensor in shape [B,H,W,C]
            frame_sample_count: Number of frames to sample
            max_pixels: Max pixels for each frame
            
        Returns:
            List of processed PIL images
        """
        # Get the batch size (number of frames)
        batch_size = images_tensor.shape[0]
        
        # Sample the frames evenly from the batch
        if batch_size <= frame_sample_count:
            # Use all frames if we have fewer than requested
            sampled_indices = list(range(batch_size))
        else:
            # Sample evenly across the frames
            sampled_indices = [int(i * (batch_size - 1) / (frame_sample_count - 1)) for i in range(frame_sample_count)]
        
        # Extract the sampled frames from the tensor
        sampled_frames = [images_tensor[i] for i in sampled_indices]
        
        # Convert to PIL images
        pil_images = []
        for frame in sampled_frames:
            # Convert tensor to numpy
            frame_np = frame.cpu().numpy()
            # Scale from [0,1] to [0,255]
            frame_np = (frame_np * 255).astype(np.uint8)
            # Convert to PIL
            pil_image = Image.fromarray(frame_np)
            
            # Calculate resize dimensions if needed
            if max_pixels > 0:
                width, height = pil_image.size
                if width * height > max_pixels:
                    scale = math.sqrt(max_pixels / (width * height))
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            pil_images.append(pil_image)
        
        return pil_images

    def analyze_sequence(self, input_mode="Frames", model_name="Qwen2.5-VL-3B-Instruct-AWQ", 
                         profile="HyVideoAnalyzer - Simple one line prompt", max_new_tokens=512, 
                         frame_sample_count=16, temperature=0.7, analysis_type="Full sequence", 
                         language="English", images=None, video_file=None, fps=8.0, max_pixels=512*512, 
                         fallback_frame_count=4, custom_system_prompt="", prefix="", suffix="", 
                         seed=-1, negative_prompt="None", model_offload="Yes", precision="float16"):
        """
        Main function to analyze video sequences and generate text prompts.
        
        Args:
            input_mode: "Frames" or "Video File"
            model_name: The name of the Qwen2.5-VL model to use
            profile: The system prompt profile to use
            max_new_tokens: Maximum number of tokens to generate
            frame_sample_count: Number of frames to sample from the input
            temperature: Temperature for text generation
            analysis_type: Type of analysis to perform
            language: Output language
            images: Input frames tensor (from LoadVideo node)
            video_file: Video file path
            fps: Frames per second for video processing
            max_pixels: Maximum pixels per frame
            fallback_frame_count: Fallback frame count if primary fails
            custom_system_prompt: Custom system prompt
            prefix: Text to add before the generated text
            suffix: Text to add after the generated text
            seed: Random seed for generation
            negative_prompt: Negative prompt configuration
            model_offload: Whether to offload model from GPU when not in use
            precision: Model precision setting
        
        Returns:
            Tuple of (sequence_description, scene_breakdown, preview_image, negative_prompt)
        """
        
        if not QWEN_AVAILABLE:
            raise ImportError("Transformers package not found. Please install with: pip install transformers torch")
        
        # Initialize sampled_indices to empty list as a fallback
        sampled_indices = []
        
        # Ensure model is loaded and on the correct device
        self.load_model(model_name, precision=precision, model_offload=model_offload)
        
        # Make sure model is on GPU for inference - this is critical since the model might have been offloaded
        if self.device == "cuda":
            logger.info("Enforcing model on CUDA for inference")
            try:
                # Force the model to CUDA, regardless of current state
                if hasattr(self.model, 'hf_device_map'):
                    logger.info("Model uses HF device map, skipping direct device movement")
                else:
                    self.model = self.model.to("cuda:0")
                    logger.info(f"Model moved to {self.model.device}")
            except Exception as e:
                logger.warning(f"Error moving model to CUDA: {e}")
        
        # Process the input frames
        if input_mode == "Frames" and images is not None:
            # Get frames from the input tensor
            pil_images = self.process_frames(images, frame_sample_count, max_pixels)
            # For Frames mode, initialize sampled_indices with sequential numbers
            # This matches what we do for Video File mode
            sampled_indices = list(range(len(pil_images)))
        elif input_mode == "Video File" and video_file is not None:
            # Load frames from video file
            try:
                # For ComfyUI compatibility - get the full path of the video file
                video_path = folder_paths.get_annotated_filepath(video_file)
                
                # We'll use qwen_vl_utils if available, otherwise fallback to a simpler method
                try:
                    from qwen_vl_utils import fetch_video
                    # Convert the video to frames
                    video_info = {
                        "video": video_path,
                        "fps": fps, 
                        "max_pixels": max_pixels
                    }
                    video_frames = fetch_video(video_info)
                    
                    # If we got a tensor, convert to PIL images
                    if isinstance(video_frames, torch.Tensor):
                        pil_images = []
                        sampled_indices = []  # Store frame indices for scene breakdown
                        for i in range(video_frames.shape[0]):
                            # Convert from TCHW to HWC format
                            frame = video_frames[i].permute(1, 2, 0)
                            
                            # Ensure values are in the [0,1] range for proper image conversion
                            # This fixes the "negative" appearance issue
                            if frame.max() > 1.0:
                                frame = frame / 255.0
                                
                            # Convert to PIL
                            frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                            pil_images.append(Image.fromarray(frame_np))
                            sampled_indices.append(i)  # Store original frame index
                    else:
                        # Already PIL images
                        pil_images = video_frames
                        sampled_indices = list(range(len(pil_images)))  # Store frame indices
                        
                    # Sample frames if needed
                    if len(pil_images) > frame_sample_count:
                        indices = [int(i * (len(pil_images) - 1) / (frame_sample_count - 1)) for i in range(frame_sample_count)]
                        # Store the actual frame indices for the breakdown
                        frame_indices = [sampled_indices[i] for i in indices]
                        pil_images = [pil_images[i] for i in indices]
                        sampled_indices = frame_indices
                    
                except ImportError:
                    # Fallback to using PIL directly
                    import av
                    container = av.open(video_path)
                    video_stream = next(s for s in container.streams if s.type == 'video')
                    
                    # Calculate frame extraction rate
                    total_frames = video_stream.frames
                    
                    if total_frames <= frame_sample_count:
                        # If there are fewer frames than requested, use all frames
                        indices = list(range(total_frames))
                    else:
                        # Sample frames evenly
                        indices = [int(i * (total_frames - 1) / (frame_sample_count - 1)) for i in range(frame_sample_count)]
                    
                    pil_images = []
                    sampled_indices = []  # Store frame indices for scene breakdown
                    for i, frame in enumerate(container.decode(video_stream)):
                        if i in indices:
                            # Convert frame to PIL
                            img = frame.to_image()
                            
                            # Resize if needed
                            if max_pixels > 0:
                                width, height = img.size
                                if width * height > max_pixels:
                                    scale = math.sqrt(max_pixels / (width * height))
                                    new_width = int(width * scale)
                                    new_height = int(height * scale)
                                    img = img.resize((new_width, new_height), Image.LANCZOS)
                            
                            pil_images.append(img)
                            sampled_indices.append(i)  # Store the original frame index
                            
                            # Break if we have enough frames
                            if len(pil_images) >= frame_sample_count:
                                break
            except Exception as e:
                logger.error(f"Error loading video: {e}")
                raise RuntimeError(f"Failed to load video: {e}")
        else:
            raise ValueError("No valid input provided. Please provide either image frames or a video file.")
        
        # Create a montage of frames for preview
        preview_image = self.create_montage(pil_images)
        
        # Prepare the system prompt
        system_prompt = self.get_system_prompt(profile, custom_system_prompt, language)
        
        # Add specific instructions based on the analysis type
        if analysis_type == "Key scenes":
            system_prompt += "\n\nProvide a breakdown of key scenes in the video, highlighting important visual elements and transitions."
        elif analysis_type == "Single summary":
            system_prompt += "\n\nProvide a single, concise summary of the entire video sequence."
        
        # Prepare the messages for the model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": []}
        ]
        
        # Add all images to the user message
        for img in pil_images:
            messages[1]["content"].append({"type": "image", "image": img})
        
        # Add the text prompt
        prompt_text = "Analyze this video sequence and provide a descriptive prompt that captures its essence."
        if language == "Chinese":
            prompt_text = "分析这个视频序列，并提供一个能够捕捉其本质的描述性提示。"
        
        messages[1]["content"].append({"type": "text", "text": prompt_text})
        
        # Process with the model
        try:
            # Format the input for the model
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Extract images and prepare inputs
            image_inputs = [item["image"] for item in messages[1]["content"] if "image" in item]
            
            # First prepare inputs without device specification
            inputs = self.processor(text=[text], images=image_inputs, return_tensors="pt")
            
            # Then forcefully move both model and inputs to same device
            if self.device == "cuda":
                logger.info("Ensuring model and inputs on same CUDA device")
                device = "cuda:0"
                try:
                    if not hasattr(self.model, 'hf_device_map'):
                        self.model = self.model.to(device)
                    inputs = inputs.to(device)
                except Exception as e:
                    logger.warning(f"Error aligning devices: {e}")
            else:
                # For CPU inference
                device = self.device
                self.model = self.model.to(device)
                inputs = inputs.to(device)
            
            # Log device information for debugging
            if hasattr(self.model, 'device'):
                logger.info(f"Model device: {self.model.device}")
            logger.info(f"Input IDs device: {inputs.input_ids.device}")
            
            # Set the seed for reproducibility
            if seed >= 0:
                torch.manual_seed(seed)
                
            # Generate the response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=temperature > 0,
                    temperature=max(0.01, temperature),  # Avoid division by zero
                    max_new_tokens=max_new_tokens,
                )
            
            # Decode the generated text
            generated_text = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            
            # Try with fewer frames as a fallback
            if frame_sample_count > fallback_frame_count and len(pil_images) > fallback_frame_count:
                logger.info(f"Retrying with {fallback_frame_count} frames instead of {frame_sample_count}")
                
                try:
                    # Sample fewer frames
                    reduced_indices = [int(i * (len(pil_images) - 1) / (fallback_frame_count - 1)) for i in range(fallback_frame_count)]
                    reduced_frames = [pil_images[i] for i in reduced_indices]
                    
                    # Update sampled_indices for the reduced set
                    if sampled_indices and len(sampled_indices) >= len(reduced_indices):
                        reduced_frame_indices = [sampled_indices[i] for i in reduced_indices]
                    else:
                        # Fallback if sampled_indices is not properly set
                        reduced_frame_indices = reduced_indices
                    
                    # Update the messages
                    messages[1]["content"] = []
                    for img in reduced_frames:
                        messages[1]["content"].append({"type": "image", "image": img})
                    messages[1]["content"].append({"type": "text", "text": prompt_text})
                    
                    # Try again with fewer frames
                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs = [item["image"] for item in messages[1]["content"] if "image" in item]
                    inputs = self.processor(text=[text], images=image_inputs, return_tensors="pt")
                    
                    # Again, ensure everything is on the same device
                    if self.device == "cuda":
                        device = "cuda:0"
                        if not hasattr(self.model, 'hf_device_map'):
                            self.model = self.model.to(device)
                        inputs = inputs.to(device)
                    else:
                        device = self.device
                        self.model = self.model.to(device)
                        inputs = inputs.to(device)
                    
                    logger.info(f"Fallback - Inputs device: {inputs.input_ids.device}")
                    
                    # Generate with reduced frame count
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            do_sample=temperature > 0,
                            temperature=max(0.01, temperature),
                            max_new_tokens=max_new_tokens,
                        )
                    
                    # Decode the generated text
                    generated_text = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                    
                    # Update sampled indices to the reduced set
                    sampled_indices = reduced_frame_indices
                    
                except Exception as e:
                    logger.error(f"Fallback also failed: {e}")
                    generated_text = f"Failed to analyze video: {str(e)}"
            else:
                generated_text = f"Failed to analyze video: {str(e)}"
        
        # Apply prefix and suffix if provided
        final_text = prefix + generated_text + suffix
        
        # Generate scene breakdown with frame indices
        scene_breakdown = self.create_scene_breakdown(final_text, sampled_indices)
        
        # Generate a negative prompt if configured
        neg_prompt_text = "None"
        if negative_prompt != "None" and negative_prompt in self.neg_prompts:
            neg_prompt_text = self.neg_prompts[negative_prompt]
        
        # Offload the model if requested
        if model_offload == "Yes" and self.device == "cuda":
            logger.info("Offloading model to CPU")
            # Only offload if we're not using a device_map
            if not hasattr(self.model, "device_map") or not self.model.device_map:
                try:
                    # Instead of moving, create a new reference to avoid device map issues
                    if hasattr(self.model, 'to'):
                        logger.info("Moving model to CPU via to() method")
                        self.model = self.model.cpu()
                except Exception as e:
                    logger.warning(f"Could not move model to CPU: {e}")
                    # Continue even if offloading fails
            
            # Force garbage collection
            torch.cuda.empty_cache()
            gc.collect()
        
        # Convert preview image to tensor for ComfyUI
        preview_tensor = self.pil_to_tensor(preview_image)
        
        return final_text, scene_breakdown, preview_tensor, neg_prompt_text

    def create_montage(self, images, max_images=9):
        """Create a montage of the sampled frames for preview."""
        if not images:
            # Create a blank image if no frames
            return Image.new("RGB", (512, 512), (0, 0, 0))
        
        # Limit the number of images to show in the montage
        images = images[:max_images]
        
        # Calculate the grid size
        grid_size = math.ceil(math.sqrt(len(images)))
        
        # Determine the size of each thumbnail
        thumb_size = 512 // grid_size
        
        # Create a new image for the montage
        montage = Image.new("RGB", (thumb_size * grid_size, thumb_size * grid_size), (0, 0, 0))
        
        # Place each image in the grid
        for i, img in enumerate(images):
            # Calculate position
            x = (i % grid_size) * thumb_size
            y = (i // grid_size) * thumb_size
            
            # Resize the image to thumbnail size
            thumb = img.copy()
            thumb.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
            
            # Calculate centered position for the thumbnail
            pos_x = x + (thumb_size - thumb.width) // 2
            pos_y = y + (thumb_size - thumb.height) // 2
            
            # Paste into montage
            montage.paste(thumb, (pos_x, pos_y))
        
        return montage

    def pil_to_tensor(self, pil_image):
        """Convert a PIL image to a torch tensor in ComfyUI format [B,H,W,C]."""
        # Convert PIL to numpy
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        
        # Convert to torch tensor and add batch dimension
        img_tensor = torch.from_numpy(img_np)[None,]
        
        return img_tensor

    def create_scene_breakdown(self, generated_text, sampled_indices=None):
        """
        Create a structured scene breakdown from the generated text.
        
        Args:
            generated_text: The text generated by the model
            sampled_indices: Optional list of frame indices used for the analysis
        """
        # In a real implementation, this could call the model again with a different prompt
        # For now, we'll just format the text into a simple scene breakdown
        
        lines = generated_text.split(". ")
        breakdown = "## Scene Breakdown\n\n"
        
        # Add information about sampled frames if available
        if sampled_indices:
            breakdown += "### Frames Used for Analysis\n"
            breakdown += "Frame indices: " + ", ".join(map(str, sampled_indices)) + "\n\n"
        
        # Create 3-5 scenes depending on the length of the text
        num_scenes = min(max(3, len(lines) // 3), 5)
        
        for i in range(num_scenes):
            start_idx = i * len(lines) // num_scenes
            end_idx = (i + 1) * len(lines) // num_scenes
            
            scene_text = ". ".join(lines[start_idx:end_idx])
            if not scene_text.endswith("."):
                scene_text += "."
                
            breakdown += f"### Scene {i+1}\n{scene_text}\n\n"
        
        return breakdown

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Return a unique hash for the inputs to determine if recomputation is needed
        # This is especially important for video file inputs
        m = hashlib.sha256()
        if kwargs.get('video_file'):
            video_path = folder_paths.get_annotated_filepath(kwargs.get('video_file'))
            if os.path.exists(video_path):
                file_hash = hashlib.sha256()
                with open(video_path, 'rb') as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        file_hash.update(byte_block)
                m.update(file_hash.digest())
        
        for k, v in kwargs.items():
            if k != 'video_file' and k != 'images':
                m.update(str(v).encode())
                
        # For image inputs, use the shape as a proxy for change detection
        if kwargs.get('images') is not None:
            images = kwargs.get('images')
            m.update(str(images.shape).encode())
            # Also add a sample of pixel values
            if images.numel() > 0:
                sample = images.flatten()[:10].tolist()
                m.update(str(sample).encode())
        
        return m.digest().hex()