import os
import soundfile as sf
import numpy as np
from typing import List, Dict

class AudioMerger:
    def __init__(self, silence_duration_ms: int = 500, chunk_silence_ms: int = 100):
        self.silence_duration_ms = silence_duration_ms
        self.chunk_silence_ms = chunk_silence_ms

    def merge(self, audio_files: List[str], output_path: str, dialogue_info: List[Dict] = None):
        """Merge multiple wav files into one with optional silence between them. Original files are NOT deleted.
        
        Args:
            audio_files: List of audio file paths
            output_path: Output file path
            dialogue_info: Optional list of dialogue metadata for each audio file to determine silence duration
        """
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
            if i < len(audio_files) - 1:
                # Determine silence duration based on whether next segment is from same dialogue
                silence_ms = self._get_silence_duration(i, dialogue_info)
                if silence_ms > 0:
                    silence_len = int(target_sr * silence_ms / 1000)
                    silence = np.zeros(silence_len, dtype=wav.dtype)
                    combined_wav.append(silence)

        if combined_wav:
            final_wav = np.concatenate(combined_wav)
            sf.write(output_path, final_wav, target_sr)
            print(f"Merged audio saved to {output_path}. Intermediate files remain in the output directory.")
        else:
            print("No audio files to merge.")
    
    def _get_silence_duration(self, current_idx: int, dialogue_info: List[Dict] = None) -> int:
        """Determine silence duration between current and next audio segment."""
        if not dialogue_info or current_idx >= len(dialogue_info) - 1:
            return self.silence_duration_ms
        
        current = dialogue_info[current_idx]
        next_item = dialogue_info[current_idx + 1]
        
        # Check if both are segments of the same original dialogue line
        if (current.get('is_segment') and next_item.get('is_segment') and
            current.get('original_line_idx') == next_item.get('original_line_idx')):
            # Same dialogue, use shorter chunk silence
            return self.chunk_silence_ms
        else:
            # Different dialogues, use longer silence
            return self.silence_duration_ms
