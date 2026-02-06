import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import threading
from typing import Dict, List

# Add parent directory to path for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dialogue_parser import DialogueParser
from text_splitter import TextSplitter
from batch_synthesizer import BatchDialogueSynthesizer
from audio_merger import AudioMerger

class DialogueTTSGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Qwen3-TTS 批量对话合成工具")
        self.root.geometry("900x700")
        
        self.parser = None
        self.speaker_widgets = {}
        
        self._create_widgets()
    
    def _create_widgets(self):
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # 1. Script Loading Section
        script_frame = ttk.LabelFrame(main_frame, text="1. 对话脚本", padding="10")
        script_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(script_frame, text="脚本文件:").grid(row=0, column=0, sticky=tk.W)
        self.script_path_var = tk.StringVar()
        ttk.Entry(script_frame, textvariable=self.script_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(script_frame, text="浏览...", command=self._browse_script).grid(row=0, column=2)
        ttk.Button(script_frame, text="加载", command=self._load_script).grid(row=0, column=3, padx=5)
        
        # 2. Speaker Configuration Section
        speaker_frame = ttk.LabelFrame(main_frame, text="2. 角色配置", padding="10")
        speaker_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        main_frame.rowconfigure(1, weight=1)
        
        # Scrollable canvas for speakers
        canvas = tk.Canvas(speaker_frame, height=200)
        scrollbar = ttk.Scrollbar(speaker_frame, orient="vertical", command=canvas.yview)
        self.speaker_container = ttk.Frame(canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        speaker_frame.columnconfigure(0, weight=1)
        speaker_frame.rowconfigure(0, weight=1)
        
        canvas.create_window((0, 0), window=self.speaker_container, anchor="nw")
        self.speaker_container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # 3. Output Settings Section
        output_frame = ttk.LabelFrame(main_frame, text="3. 输出设置", padding="10")
        output_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(output_frame, text="输出目录:").grid(row=0, column=0, sticky=tk.W)
        self.output_dir_var = tk.StringVar(value="./output_dialogue")
        ttk.Entry(output_frame, textvariable=self.output_dir_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(output_frame, text="浏览...", command=self._browse_output_dir).grid(row=0, column=2)
        
        self.merge_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="合并为单个文件", variable=self.merge_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        ttk.Label(output_frame, text="对话间隔(ms):").grid(row=2, column=0, sticky=tk.W)
        self.silence_var = tk.IntVar(value=500)
        ttk.Spinbox(output_frame, from_=0, to=5000, textvariable=self.silence_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(output_frame, text="句子间隔(ms):").grid(row=3, column=0, sticky=tk.W)
        self.chunk_silence_var = tk.IntVar(value=100)
        ttk.Spinbox(output_frame, from_=0, to=2000, textvariable=self.chunk_silence_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(output_frame, text="文本拆分长度:").grid(row=4, column=0, sticky=tk.W)
        self.max_chars_var = tk.IntVar(value=100)
        ttk.Spinbox(output_frame, from_=50, to=1000, textvariable=self.max_chars_var, width=10).grid(row=4, column=1, sticky=tk.W, padx=5)
        
        # 4. Model Settings Section
        model_frame = ttk.LabelFrame(main_frame, text="4. 模型设置", padding="10")
        model_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(model_frame, text="模型路径:").grid(row=0, column=0, sticky=tk.W)
        self.model_path_var = tk.StringVar(value="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        ttk.Entry(model_frame, textvariable=self.model_path_var, width=50).grid(row=0, column=1, padx=5)
        
        ttk.Label(model_frame, text="设备:").grid(row=1, column=0, sticky=tk.W)
        self.device_var = tk.StringVar(value="cuda:0")
        ttk.Combobox(model_frame, textvariable=self.device_var, values=["cuda:0", "cuda:1", "cpu"], width=15).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # 5. Synthesis Control Section
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.start_button = ttk.Button(control_frame, text="🚀 开始合成", command=self._start_synthesis)
        self.start_button.grid(row=0, column=0, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100, length=400)
        self.progress_bar.grid(row=1, column=0, pady=5)
        
        self.status_label = ttk.Label(control_frame, text="就绪")
        self.status_label.grid(row=2, column=0)
        
        # 6. Log Section
        log_frame = ttk.LabelFrame(main_frame, text="日志", padding="10")
        log_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        main_frame.rowconfigure(5, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, state='disabled')
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
    
    def _log(self, message):
        """Add message to log."""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')
    
    def _browse_script(self):
        filename = filedialog.askopenfilename(
            title="选择对话脚本",
            filetypes=[("支持的格式", "*.json *.txt"), ("JSON 文件", "*.json"), ("TXT 文件", "*.txt"), ("所有文件", "*.*")]
        )
        if filename:
            self.script_path_var.set(filename)
    
    def _browse_output_dir(self):
        dirname = filedialog.askdirectory(title="选择输出目录")
        if dirname:
            self.output_dir_var.set(dirname)
    
    def _load_script(self):
        script_path = self.script_path_var.get()
        if not script_path or not os.path.exists(script_path):
            messagebox.showerror("错误", "请选择有效的脚本文件")
            return
        
        try:
            self.parser = DialogueParser(script_path)
            self.parser.validate(skip_audio_check=True)
            
            roles = self.parser.get_roles()
            self._create_speaker_widgets(roles)
            
            self._log(f"成功加载脚本: {script_path}")
            self._log(f"检测到 {len(roles)} 个角色: {', '.join(roles)}")
            
        except Exception as e:
            messagebox.showerror("加载失败", str(e))
            self._log(f"错误: {e}")
    
    def _create_speaker_widgets(self, roles: List[str]):
        """Create input widgets for each speaker."""
        # Clear existing widgets
        for widget in self.speaker_container.winfo_children():
            widget.destroy()
        self.speaker_widgets.clear()
        
        # Create header
        ttk.Label(self.speaker_container, text="角色", font=('', 9, 'bold')).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(self.speaker_container, text="参考音频", font=('', 9, 'bold')).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(self.speaker_container, text="参考文本", font=('', 9, 'bold')).grid(row=0, column=2, padx=5, pady=5, columnspan=2)
        
        # Create row for each role
        for i, role in enumerate(roles, start=1):
            ttk.Label(self.speaker_container, text=role).grid(row=i, column=0, padx=5, pady=2, sticky=tk.W)
            
            # Audio file selection
            audio_var = tk.StringVar()
            audio_frame = ttk.Frame(self.speaker_container)
            audio_frame.grid(row=i, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))
            
            audio_entry = ttk.Entry(audio_frame, textvariable=audio_var, width=25)
            audio_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            browse_audio_btn = ttk.Button(
                audio_frame, 
                text="...", 
                width=3,
                command=lambda v=audio_var, tv=None: self._browse_audio(v, tv)
            )
            browse_audio_btn.pack(side=tk.LEFT, padx=2)
            
            # Reference text input and file selection
            text_var = tk.StringVar()
            text_frame = ttk.Frame(self.speaker_container)
            text_frame.grid(row=i, column=2, padx=5, pady=2, sticky=(tk.W, tk.E))
            
            text_entry = ttk.Entry(text_frame, textvariable=text_var, width=25)
            text_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            browse_text_btn = ttk.Button(
                text_frame,
                text="...",
                width=3,
                command=lambda tv=text_var: self._browse_ref_text(tv)
            )
            browse_text_btn.pack(side=tk.LEFT, padx=2)
            
            # Update browse_audio_btn command to pass text_var
            browse_audio_btn.config(command=lambda v=audio_var, tv=text_var: self._browse_audio(v, tv))
            
            # Language selection
            lang_var = tk.StringVar(value="Chinese")
            ttk.Combobox(
                self.speaker_container, 
                textvariable=lang_var, 
                values=["Chinese", "English", "Japanese", "Korean", "Auto"],
                width=10
            ).grid(row=i, column=3, padx=5, pady=2)
            
            self.speaker_widgets[role] = {
                "audio": audio_var,
                "text": text_var,
                "language": lang_var
            }
    
    def _browse_audio(self, audio_var, text_var=None):
        """Browse for audio file and auto-load corresponding text file."""
        filename = filedialog.askopenfilename(
            title="选择参考音频",
            filetypes=[("音频文件", "*.wav *.mp3 *.flac"), ("所有文件", "*.*")]
        )
        if filename:
            audio_var.set(filename)
            
            # Auto-load corresponding text file if exists
            if text_var is not None:
                base_path = os.path.splitext(filename)[0]
                txt_path = base_path + ".txt"
                
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            ref_text = f.read().strip()
                        text_var.set(ref_text)
                        self._log(f"自动加载参考文本: {os.path.basename(txt_path)}")
                    except Exception as e:
                        self._log(f"加载参考文本失败: {e}")
    
    def _browse_ref_text(self, text_var):
        """Browse for reference text file."""
        filename = filedialog.askopenfilename(
            title="选择参考文本文件",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    ref_text = f.read().strip()
                text_var.set(ref_text)
                self._log(f"加载参考文本: {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("错误", f"无法读取文本文件: {e}")
                self._log(f"加载参考文本失败: {e}")
    
    def _start_synthesis(self):
        """Start synthesis in a separate thread."""
        if not self.parser:
            messagebox.showerror("错误", "请先加载脚本")
            return
        
        # Validate speaker configs
        for role, widgets in self.speaker_widgets.items():
            audio_path = widgets["audio"].get()
            if not audio_path or not os.path.exists(audio_path):
                messagebox.showerror("错误", f"角色 '{role}' 的参考音频无效")
                return
        
        # Update parser with speaker configs
        for role, widgets in self.speaker_widgets.items():
            self.parser.update_speaker_config(
                role=role,
                ref_audio=widgets["audio"].get(),
                ref_text=widgets["text"].get(),
                language=widgets["language"].get()
            )
        
        # Disable start button
        self.start_button.config(state='disabled')
        self.progress_var.set(0)
        self._log("\n=== 开始合成 ===")
        
        # Run synthesis in thread
        thread = threading.Thread(target=self._synthesis_worker)
        thread.daemon = True
        thread.start()
    
    def _synthesis_worker(self):
        """Worker thread for synthesis."""
        try:
            # Get dialogues and split
            dialogues = self.parser.get_dialogues()
            splitter = TextSplitter(max_chars=self.max_chars_var.get())
            processed_dialogues = splitter.process_dialogues(dialogues)
            
            self._log(f"处理后共 {len(processed_dialogues)} 段对话")
            
            # Initialize synthesizer
            self._log("初始化模型...")
            synthesizer = BatchDialogueSynthesizer(
                model_path=self.model_path_var.get(),
                device=self.device_var.get()
            )
            
            # Prepare speakers
            speakers = self.parser.get_speakers()
            synthesizer.prepare_speakers(speakers)
            
            # Synthesize
            self._log("开始合成音频...")
            output_dir = self.output_dir_var.get()
            metadata = self.parser.get_metadata()
            
            generated_files, dialogue_info = synthesizer.synthesize(
                processed_dialogues,
                output_dir,
                default_lang=metadata.get("default_language", "Chinese")
            )
            
            self.progress_var.set(80)
            
            # Merge if requested
            if self.merge_var.get() and generated_files:
                self._log("合并音频文件...")
                merger = AudioMerger(
                    silence_duration_ms=self.silence_var.get(),
                    chunk_silence_ms=self.chunk_silence_var.get()
                )
                output_name = metadata.get("title", "combined_dialogue").replace(" ", "_") + ".wav"
                output_path = os.path.join(output_dir, output_name)
                merger.merge(generated_files, output_path, dialogue_info=dialogue_info)
            
            self.progress_var.set(100)
            self._log(f"\n✓ 合成完成! 输出目录: {os.path.abspath(output_dir)}")
            
            self.root.after(0, lambda: messagebox.showinfo("完成", "音频合成完成!"))
            
        except Exception as e:
            self._log(f"\n✗ 错误: {e}")
            self.root.after(0, lambda: messagebox.showerror("合成失败", str(e)))
        
        finally:
            self.root.after(0, lambda: self.start_button.config(state='normal'))

def main():
    root = tk.Tk()
    app = DialogueTTSGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
