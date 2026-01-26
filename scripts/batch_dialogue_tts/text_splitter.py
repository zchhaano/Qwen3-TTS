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
                
                # If a single sentence is still too long, split by word boundaries
                if len(s) > self.max_chars:
                    chunks.extend(self._split_by_words(s))
                    current_chunk = ""
                else:
                    current_chunk = s
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def _split_by_words(self, text: str) -> List[str]:
        """Split long text by word boundaries to avoid cutting words in the middle."""
        chunks = []
        words = re.split(r'(\s+)', text)  # Split by whitespace, keeping delimiters
        
        current_chunk = ""
        for word in words:
            # If adding this word would exceed max_chars
            if len(current_chunk) + len(word) > self.max_chars:
                if current_chunk:
                    chunks.append(current_chunk.rstrip())
                    current_chunk = word.lstrip() if not word.isspace() else ""
                else:
                    # Single word is longer than max_chars, must split it
                    # But try to split at hyphens first (for compound words)
                    if '-' in word and len(word) > self.max_chars:
                        parts = word.split('-')
                        for i, part in enumerate(parts):
                            if i < len(parts) - 1:
                                part += '-'
                            if len(current_chunk) + len(part) <= self.max_chars:
                                current_chunk += part
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk)
                                current_chunk = part
                    else:
                        # Last resort: hard split but warn
                        chunks.append(word[:self.max_chars])
                        current_chunk = word[self.max_chars:]
            else:
                current_chunk += word
        
        if current_chunk:
            chunks.append(current_chunk.rstrip())
        
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
