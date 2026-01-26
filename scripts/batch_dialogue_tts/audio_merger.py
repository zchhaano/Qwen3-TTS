import os
import soundfile as sf
import numpy as np
from typing import List

class AudioMerger:
    def __init__(self, silence_duration_ms: int = 500):
        self.silence_duration_ms = silence_duration_ms

    def merge(self, audio_files: List[str], output_path: str):
        """Merge multiple wav files into one with optional silence between them. Original files are NOT deleted."""
        print(f"Merging {len(audio_files)} files into {output_path}...")
        
        combined_wav = []
        target_sr = None

        for i, file_path in enumerate(audio_files):
            if not os.path.exists(file_path):
                print(f"Warning: Audio file {file_path} not found, skipping merge.")
                continue
                
            wav, sr = sf.read(file_path)
            
            if target_sr is None:
                target_sr = sr
            elif sr != target_sr:
                print(f"Warning: Sample rate mismatch in {file_path}. Expected {target_sr}, got {sr}")
            
            combined_wav.append(wav)
            
            # Add silence between files (except after the last one)
            if i < len(audio_files) - 1 and self.silence_duration_ms > 0:
                silence_len = int(target_sr * self.silence_duration_ms / 1000)
                silence = np.zeros(silence_len, dtype=wav.dtype)
                combined_wav.append(silence)

        if combined_wav:
            final_wav = np.concatenate(combined_wav)
            sf.write(output_path, final_wav, target_sr)
            print(f"Merged audio saved to {output_path}. Intermediate files remain in the output directory.")
        else:
            print("No audio files to merge.")
