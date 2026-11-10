# Handoff: set up the Wan video generator on another computer

Everything runs locally, so nothing carries over from an account: you copy files and install the tools below. The
scripts expect the same paths as here, so keep them: the code at `C:\wan-video-studio`, ComfyUI at `%USERPROFILE%\ComfyUI`.

## What the new computer needs

- Windows with an NVIDIA card. This was built on an RTX 3070 Laptop with 8 GB of GPU memory and 31 GB of RAM: less GPU
  memory won't run the 14B models, and more makes everything faster.
- About 100 GB free: the models are 89 GB.
- Python 3.12, an NVIDIA driver that supports CUDA 13, and Git.

## Quickest way: copy two folders

If both computers are yours, copy these with a USB drive or over the network, then skip to step 5:

1. `C:\wan-video-studio` (the code; it has no online copy yet, so this is the only way to get it).
2. `C:\Users\<you>\ComfyUI` (ComfyUI, its add-on, the Python environment and all models).
   The `venv` inside may not work on a new computer if Python is installed in a different place: if
   `ComfyUI\venv\Scripts\python.exe --version` fails, delete `venv` and do step 3.

## Fresh install

1. **Code:** copy `C:\wan-video-studio` to the same place.
2. **ComfyUI:** `git clone https://github.com/comfyanonymous/ComfyUI "%USERPROFILE%\ComfyUI"` (this setup used commit 2c75dc1).
3. **Python environment**, in `%USERPROFILE%\ComfyUI`:

   ```bash
   python -m venv venv
   ```
   ```bash
   venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```
   ```bash
   venv/Scripts/python.exe -m pip install -r requirements.txt av gguf protobuf scipy yt-dlp
   ```

   Versions used here: torch 2.14.1, torchvision 0.29.1, av 19.0.2, gguf 0.19.0.
4. **Add-on:** `git clone https://github.com/city96/ComfyUI-GGUF "%USERPROFILE%\ComfyUI\custom_nodes\ComfyUI-GGUF"`
5. **Models** in `%USERPROFILE%\ComfyUI\models` (all from Hugging Face):

   | Folder | Files | From |
   |---|---|---|
   | `unet` | `Wan2.2-VACE-Fun-A14B-high-noise-Q6_k.gguf`, `Wan2.2-VACE-Fun-A14B-low-noise-Q6_K.gguf` | QuantStack/Wan2.2-VACE-Fun-A14B-GGUF |
   | `diffusion_models` | `wan2.2_ti2v_5B_fp16`, `wan2.2_t2v_high_noise_14B_fp8_scaled`, `wan2.2_t2v_low_noise_14B_fp8_scaled` (.safetensors) | Comfy-Org/Wan_2.2_ComfyUI_Repackaged |
   | `diffusion_models` | `wan_animate_2_distill_int8_convrot.safetensors` | Comfy-Org/Wan-Animate-2 |
   | `vae` | `wan2.2_vae`, `wan_2.1_vae` (.safetensors) | Comfy-Org/Wan_2.2_ComfyUI_Repackaged |
   | `vae` | `Wan2_1_VAE_bf16.safetensors` | Comfy-Org/Wan-Animate-2 |
   | `text_encoders` | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | Comfy-Org/Wan_2.2_ComfyUI_Repackaged |
   | `clip_vision` | `clip_vision_h.safetensors` | Comfy-Org/Wan-Animate-2 |
   | `loras` | `Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors` | hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF |
   | `upscale_models` | `RealESRGAN_x4plus.safetensors` | Comfy-Org/Real-ESRGAN_repackaged |
   | `frame_interpolation` | `film_net_fp16.safetensors` | Comfy-Org/Frame-Interpolation |

   The two VACE files and the `wan2.2_ti2v_5B` file are all the one-prompt chain needs; the rest are for other modes.

6. **Blender 5.2** installed at the default `C:\Program Files\Blender Foundation\Blender 5.2` (it renders the motion).
   Motion capture files download by themselves on first use.
7. **Optional, Ollama** with `ollama pull llama3.2`: only for the chat on the Studio page and `--auto-reference`.

## Start it

Double-click `C:\wan-video-studio\Start Wan Studio.bat`. It starts ComfyUI (http://127.0.0.1:8188) and the Studio page
(http://127.0.0.1:8190, progress at /progress). Running `start_studio.ps1` directly is blocked by Windows' default
script setting; the .bat gets around that for this one script.

Check it works with a quick 5B clip (about 2 minutes):

```bash
cd /c/wan-video-studio && ~/ComfyUI/venv/Scripts/python.exe generate_video.py "a dog running on a beach"
```

Then the full chain (about 2 hours at 832x480, about 15 with `--full`):

```bash
cd /c/wan-video-studio && ~/ComfyUI/venv/Scripts/python.exe auto_vace.py "a woman dancing salsa in a ballroom"
```

Clips are saved to `%USERPROFILE%\ComfyUI\output\video`.

## Claude Code on the new computer

To send one prompt and have Claude run the whole chain, copy the folders in `C:\wan-video-studio\skills` into
`%USERPROFILE%\.claude\skills`. `vace-video` is the main one. Claude's memory of how you like to work doesn't carry over,
so tell it once: "when I send a video prompt, run the vace-video skill without asking about settings".

## Good to know

- Times above are for the RTX 3070 Laptop; a bigger card is much faster.
- `auto_vace.py` keeps the computer awake while it runs, but a Windows Update restart or a closed lid still stops it.
  Pause updates during long runs.
- Your past clips, ratings and sessions are in `ComfyUI\output`; copy that folder too if you want them.
- More detail on every option is in `README.md`.
