# ComfyUI-IF_VideoPrompts

A ComfyUI extension that provides video sequence analysis and prompting using advanced multimodal LLMs. This extension uses the Qwen2.5-VL models from Alibaba to analyze video sequences and generate detailed descriptions.

## Important Requirements

This extension requires **transformers 4.49.0 or above** to work properly. Earlier versions (including 4.48.0) will cause errors.

## Features

- **Video Frame Analysis**: Analyze a sequence of video frames loaded via the Video Helper Suite nodes
- **Direct Video File Processing**: Process MP4 and other video files directly without pre-loading frames
- **Multiple Analysis Types**:
  - Full sequence narratives
  - Key scene breakdowns
  - Single summaries
- **Language Support**: English and Chinese output
- **Customizable Prompting**: Define your own system prompts or use provided presets
- **Negative Prompt Generation**: Generate negative prompts for video content

## Installation

### Method 1: Using the installation script (Recommended)

This method handles dependency conflicts automatically:

1. Clone this repository into your ComfyUI `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-IF_VideoPrompts.git
```

2. Run the installation script:
```bash
cd ComfyUI-IF_VideoPrompts
python install.py
```

3. Restart ComfyUI

### Method 2: Manual installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-IF_VideoPrompts.git
```

2. Install required dependencies:
```bash
# First uninstall potentially conflicting packages
pip uninstall -y autoawq transformers

# Install specific transformers version
pip install transformers==4.49.0

# Install compatible autoawq version WITHOUT dependencies to prevent transformers downgrade
# If you want to use AWQ to save VRAM and up to 3x faster inference
# you need to install triton and autoawq



# Then install other dependencies
pip install -r requirements.txt
```

I also have precompiled wheels for FA2 sageattention and trton for windows 10 for cu126 and pytorch 2.6.3 and python 12+ https://huggingface.co/impactframes/ComfyUI_desktop_wheels_win_cp12_cu126/tree/main



3. Restart ComfyUI

### Method 3: Direct pip installation

If you want to install the dependencies directly with pip:

```bash
pip install transformers==4.49.0 opencv-python decord huggingface_hub pillow torch numpy tokenizers safetensors accelerate tqdm psutil packaging
pip install --no-deps autoawq==0.2.8
```

### Dependency Conflicts

If you encounter dependency conflicts (especially with transformers and autoawq), try:

```bash
# Uninstall problematic packages
pip uninstall -y autoawq transformers

# Install specific transformers version
pip install transformers==4.49.0

# Then install autoawq WITHOUT dependencies
pip install --no-deps autoawq==0.2.8
```

## Usage

### Frame-based Mode

1. Load a video using the VideoHelperSuite's LoadVideo node
2. Connect the output to the VideoSequenceAnalyzer node
3. Select "Frames" as the input mode
4. Choose your preferred model, analysis type, and other settings
5. Run the workflow to get a detailed description of the video sequence

### Direct Video File Mode

1. Upload a video file to your ComfyUI input directory
2. Add the VideoSequenceAnalyzer node
3. Select "Video File" as the input mode
4. Choose your video file from the dropdown
5. Configure FPS, analysis type, and other settings
6. Run the workflow to get a detailed description of the video

## Models

The extension supports the following Qwen2.5-VL models:

- Qwen2.5-VL-3B-Instruct
- Qwen2.5-VL-7B-Instruct
- Qwen2.5-VL-14B-Instruct
- Qwen2.5-VL-72B-Instruct
- Qwen2.5-VL-3B-Instruct-AWQ (quantized)
- Qwen2.5-VL-7B-Instruct-AWQ (quantized)
- Qwen2.5-VL-14B-Instruct-AWQ (quantized)
- Qwen2.5-VL-72B-Instruct-AWQ (quantized)

AWQ quantized models are recommended for better performance.

## Custom Presets

You can define your own presets by adding them to the `presets/profiles.json` file.

## Troubleshooting

### "Image features and image tokens do not match" Error

If you encounter this error, try the following:
1. Switch to "Video File" input mode to use the native Qwen-VL video processing
2. Reduce the number of frames in your sequence
3. Try a different model (AWQ versions often work better)

### Transformers Version Conflicts

This extension requires transformers version 4.49.0 or higher. Earlier versions (including 4.48.0) will not work.

If autoawq or other packages downgrade your transformers version, follow these steps:

```bash
# Uninstall both packages
pip uninstall -y autoawq transformers

# Install specific transformers version first
pip install transformers==4.49.0

# Then install compatible autoawq WITHOUT dependencies
pip install --no-deps autoawq==0.2.8
```

if you have some fuckery with the LD_libray whatever do:
```
pip uninstall bitsandbytes -y
pip install bitsandbytes
```
### Missing Dependencies

If you're missing dependencies, use the installation script:
```bash
python install.py
```

## Credits

This extension uses the following components:
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) models from Alibaba
- [Video Helper Suite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) for frame extraction 


Support
If you find this tool useful, please consider supporting my work by:

Starring the repository on GitHub: ComfyUI-IF_VideoPrompts
Subscribing to my YouTube channel: Impact Frames
Follow me on X: Impact Frames X
Thank You!
<img src="https://count.getloli.com/get/@IFAIVideoPrompts_comfy?theme=moebooru" alt=":IFAIVideoPrompts_comfy" />
