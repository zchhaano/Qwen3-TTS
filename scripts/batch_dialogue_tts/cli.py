import argparse
import os
import sys
import torch
from .dialogue_parser import DialogueParser
from .text_splitter import TextSplitter
from .batch_synthesizer import BatchDialogueSynthesizer
from .audio_merger import AudioMerger

def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Batch Dialogue Synthesis Tool (JSON/TXT)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to dialogue script (JSON or TXT)")
    parser.add_argument("--output-dir", "-o", type=str, default="./output_dialogue", help="Output directory for audio files")
    parser.add_argument("--model-path", "-m", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="Path to Qwen3-TTS-Base model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    parser.add_argument("--merge", action="store_true", help="Merge generated clips into one file (original clips are kept)")
    parser.add_argument("--silence", type=int, default=500, help="Silence duration (ms) between clips when merging")
    parser.add_argument("--max-chars", type=int, default=100, help="Max characters per synthesis segment (default 100)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        sys.exit(1)

    # 1. Parse Script
    print("Step 1: Parsing dialogue script...")
    parser_obj = DialogueParser(args.input)
    
    # For TXT format, skip audio check initially (will be configured via GUI or manually)
    is_txt = args.input.lower().endswith('.txt')
    parser_obj.validate(skip_audio_check=is_txt)
    
    if is_txt:
        print("Note: TXT format detected. Make sure to configure speaker audio via GUI or update the script.")
    
    metadata = parser_obj.get_metadata()
    speakers = parser_obj.get_speakers()
    dialogues = parser_obj.get_dialogues()
    
    default_lang = metadata.get("default_language", "Chinese")
    
    # 2. Split text if needed
    print(f"Step 2: Splitting long dialogue lines (Max {args.max_chars} chars)...")
    splitter = TextSplitter(max_chars=args.max_chars)
    processed_dialogues = splitter.process_dialogues(dialogues)
    
    print(f"Total lines to process after splitting: {len(processed_dialogues)}")

    # 3. Synchronize synthesis
    print("Step 3: Initializing synthesizer and preparing voice prompts...")
    synthesizer = BatchDialogueSynthesizer(
        model_path=args.model_path,
        device=args.device
    )
    synthesizer.prepare_speakers(speakers)
    
    print("Step 4: Starting batch synthesis...")
    generated_files = synthesizer.synthesize(
        processed_dialogues, 
        args.output_dir,
        default_lang=default_lang
    )

    # 4. Merge if requested
    if args.merge and generated_files:
        print("Step 5: Merging audio files into single track...")
        merger = AudioMerger(silence_duration_ms=args.silence)
        output_name = metadata.get("title", "combined_dialogue").replace(" ", "_") + ".wav"
        output_path = os.path.join(args.output_dir, output_name)
        merger.merge(generated_files, output_path)

    print("\nBatch synthesis completed successfully!")
    print(f"Intermediate segments and final result (if merged) are in: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
