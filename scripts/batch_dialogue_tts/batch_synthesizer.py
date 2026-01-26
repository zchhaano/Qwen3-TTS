import os
import torch
import soundfile as sf
import numpy as np
from typing import List, Dict, Any, Tuple
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

class BatchDialogueSynthesizer:
    def __init__(
        self, 
        model_path: str, 
        device: str = "cuda:0", 
        dtype: torch.dtype = torch.bfloat16
    ):
        print(f"Loading model from {model_path}...")
        self.tts = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else None
        )
        self.prompts: Dict[str, List[VoiceClonePromptItem]] = {}

    def prepare_speakers(self, speakers: Dict[str, Any]):
        """Create and cache voice clone prompts for all speakers."""
        for name, info in speakers.items():
            print(f"Preparing voice prompt for speaker: {name}")
            ref_audio = info["ref_audio"]
            ref_text = info.get("ref_text")
            
            prompt = self.tts.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=False if ref_text else True
            )
            self.prompts[name] = prompt

    def synthesize(
        self, 
        dialogues: List[Dict[str, Any]], 
        output_dir: str,
        default_lang: str = "Chinese"
    ) -> List[str]:
        """Synthesize each dialogue line and save to wav files. Intermediate segments are preserved."""
        os.makedirs(output_dir, exist_ok=True)
        generated_files = []

        for i, line in enumerate(dialogues):
            role = line["role"]
            text = line["text"]
            lang = line.get("language") or default_lang
            
            print(f"[{i+1}/{len(dialogues)}] Generating audio for {role}: {text[:30]}...")
            
            # Use cached prompt
            prompt = self.prompts.get(role)
            if not prompt:
                raise ValueError(f"No prompt prepared for role: {role}")

            try:
                wavs, sr = self.tts.generate_voice_clone(
                    text=text,
                    language=lang,
                    voice_clone_prompt=prompt
                )
                
                # Filename logic: include index and role. If it's a segment, add seg suffix.
                if line.get("is_segment"):
                    # Use original line index if possible, but for simplicity we use loop index
                    # Actually, a better way would be to track original indices.
                    # For now, i is unique enough.
                    filename = f"{i:04d}_{role}_part{line.get('segment_idx', 0)}.wav"
                else:
                    filename = f"{i:04d}_{role}.wav"
                
                output_path = os.path.join(output_dir, filename)
                sf.write(output_path, wavs[0], sr)
                generated_files.append(output_path)
            except Exception as e:
                print(f"Error synthesizing line {i}: {e}")
        
        return generated_files
