import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw
import imageio
import tempfile
import os
import subprocess
import shutil

# ── gTTS Voice ──
try:
    from gtts import gTTS
    HAS_TTS = True
except:
    HAS_TTS = False

def generate_voice(text, voice_type, tmp_dir):
    if not HAS_TTS or voice_type == "none" or not text:
        return None
    try:
        mp3 = os.path.join(tmp_dir, "voice.mp3")
        wav = os.path.join(tmp_dir, "voice.wav")
        gTTS(text=text[:400], lang='en', slow=(voice_type=="neutral")).save(mp3)
        subprocess.run(["ffmpeg","-y","-i",mp3,wav], capture_output=True, check=True)
        return wav
    except Exception as e:
        print(f"TTS error: {e}")
        return None

# ── SadTalker (real talking face) ──
SADTALKER_PATH = "/app/SadTalker"

def run_sadtalker(photo_path, audio_path, output_dir):
    try:
        cmd = [
            "python", "inference.py",
            "--driven_audio", audio_path,
            "--source_image", photo_path,
            "--result_dir", output_dir,
            "--still",
            "--preprocess", "full",
            "--enhancer", "gfpgan"
        ]
        subprocess.run(cmd, check=True, cwd=SADTALKER_PATH, capture_output=True)
        files = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]
        if files:
            return os.path.join(output_dir, files[0])
    except Exception as e:
        print(f"SadTalker error: {e}")
    return None

# ── Fallback motion pipeline (if SadTalker unavailable) ──
def apply_motion(frame_array, i, total, style):
    h, w = frame_array.shape[:2]
    p = i / max(total-1, 1)
    if style == "cinematic":
        scale = 1.0 + 0.05 * p
    elif style == "anime":
        return np.roll(frame_array, int(w*0.03*np.sin(p*np.pi)), axis=1)
    else:
        scale = 1.0 + 0.025 * np.sin(p * np.pi)
    ch, cw = int(h/scale), int(w/scale)
    y1, x1 = (h-ch)//2, (w-cw)//2
    return np.array(Image.fromarray(frame_array[y1:y1+ch,x1:x1+cw]).resize((w,h),Image.LANCZOS))

def apply_style(arr, style):
    img = arr.astype(np.float32)
    if style == "cinematic":
        img[:,:,0] *= 1.12; img[:,:,2] *= 0.82
    elif style == "anime":
        img = np.clip(img*0.85+22,0,255)
    elif style == "cartoon":
        img = (img//32)*32
    return np.clip(img,0,255).astype(np.uint8)

def lip_sim(arr, i, active):
    if not active: return arr
    h, w = arr.shape[:2]
    my, mx = int(h*0.72), w//2
    r = int(w*0.03)
    op = abs(np.sin(i*0.6))*r
    f = arr.copy()
    for dy in range(-int(op), int(op)+1):
        for dx in range(-r, r+1):
            if op > 0 and dy**2/max(op**2,1)+dx**2/max(r**2,1) <= 1:
                py, px = my+dy, mx+dx
                if 0<=py<h and 0<=px<w:
                    f[py,px] = [35,15,15]
    return f

def fallback_video(photo, prompt, style, duration, voice, tmp_dir):
    fps = 24
    frames_n = int(duration) * fps
    img = np.array(Image.fromarray(photo).convert("RGB").resize((512,512),Image.LANCZOS))
    raw = os.path.join(tmp_dir, "raw.mp4")
    writer = imageio.get_writer(raw, fps=fps, codec='libx264', quality=7)
    speech = voice != "none"
    for i in range(frames_n):
        f = apply_motion(img, i, frames_n, style)
        f = apply_style(f, style)
        if speech: f = lip_sim(f, i, True)
        if i < fps*2:
            try:
                pil = Image.fromarray(f)
                ImageDraw.Draw(pil).text((14,14), prompt[:60], fill=(255,255,255))
                f = np.array(pil)
            except: pass
        writer.append_data(f)
    writer.close()
    return raw

def merge_av(video, audio, out):
    try:
        subprocess.run([
            "ffmpeg","-y","-i",video,"-i",audio,
            "-c:v","copy","-c:a","aac","-shortest",out
        ], check=True, capture_output=True)
        return out
    except:
        return video

# ── Main function ──
def generate_video(photo, prompt, style, duration, voice):
    if photo is None:
        return None, "❌ No photo uploaded."
    if not prompt or not prompt.strip():
        return None, "❌ No prompt entered."

    tmp = tempfile.mkdtemp()
    photo_path = os.path.join(tmp, "input.png")
    out_dir    = os.path.join(tmp, "out")
    final_path = os.path.join(tmp, "final.mp4")
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(photo).save(photo_path)

    audio_path = generate_voice(prompt, voice, tmp)

    # Try SadTalker first
    if audio_path and os.path.exists(SADTALKER_PATH):
        result = run_sadtalker(photo_path, audio_path, out_dir)
        if result:
            return result, f"✅ Real talking face! ({duration}s · {style} · voice: {voice})"

    # Fallback
    raw = fallback_video(photo, prompt, style, duration, voice, tmp)
    if audio_path and os.path.exists(audio_path):
        final = merge_av(raw, audio_path, final_path)
    else:
        final = raw

    return final, f"✅ Done! ({duration}s · {style} · voice: {voice})"

# ── Gradio UI ──
with gr.Blocks(
    title="Menace AI V2",
    css="""
        body { background: #0a0010 !important; }
        .gradio-container { background: #0a0010 !important; }
        h1 { color: #a78bfa !important; }
    """
) as demo:
    gr.Markdown("## 👿 Menace AI V2")
    with gr.Row():
        photo_in  = gr.Image(label="Photo", type="numpy")
        prompt_in = gr.Textbox(label="Prompt", lines=4, placeholder="What should happen / what should they say...")
    with gr.Row():
        style_in    = gr.Dropdown(["realistic","cinematic","anime","cartoon"], value="realistic", label="Style")
        duration_in = gr.Dropdown(["3","5","10"], value="5", label="Duration (s)")
        voice_in    = gr.Dropdown(["none","male","female","neutral"], value="female", label="Voice")
    btn        = gr.Button("👿 Generate Video", variant="primary")
    video_out  = gr.Video(label="Output")
    status_out = gr.Textbox(label="Status")

    btn.click(
        fn=generate_video,
        inputs=[photo_in, prompt_in, style_in, duration_in, voice_in],
        outputs=[video_out, status_out]
    )

demo.launch()
