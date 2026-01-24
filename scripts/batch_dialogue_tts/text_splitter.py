import re
from typing import List, Dict, Any

class TextSplitter:
    def __init__(self, max_chars: int = 100):
        self.max_chars = max_chars
        # Sentence boundary markers for various languages
        self.boundary_pattern = re.compile(r'([。！？！？.!?]+)')

    def split_text(self, text: str) -> List[str]:
        if len(text) <= self.max_chars:
            return [text]

        # First, try to split by sentence boundaries
        parts = self.boundary_pattern.split(text)
        
        # Re-combine parts into chunks within max_chars
        sentences = []
        for i in range(0, len(parts), 2):
            sentence = parts[i]
            if i + 1 < len(parts):
                sentence += parts[i+1]
            if sentence.strip():
                sentences.append(sentence)

        chunks = []
        current_chunk = ""
        for s in sentences:
            if len(current_chunk) + len(s) <= self.max_chars:
                current_chunk += s
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If a single sentence is still too long, hard split it
                if len(s) > self.max_chars:
                    while len(s) > self.max_chars:
                        chunks.append(s[:self.max_chars])
                        s = s[self.max_chars:]
                    current_chunk = s
                else:
                    current_chunk = s
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def process_dialogues(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed = []
        for line in dialogues:
            text = line["text"]
            if len(text) > self.max_chars:
                chunks = self.split_text(text)
                for i, chunk in enumerate(chunks):
                    new_line = line.copy()
                    new_line["text"] = chunk
                    new_line["is_segment"] = True
                    new_line["segment_idx"] = i
                    processed.append(new_line)
            else:
                processed.append(line)
        return processed
