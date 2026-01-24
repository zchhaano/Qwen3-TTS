# Qwen3-TTS 批量对话合成工具

这是一个基于 Qwen3-TTS 的多人对话批量合成工具,支持 JSON 和 TXT 两种脚本格式。

## 功能特性

- ✅ 支持 JSON 和 TXT 两种脚本格式
- ✅ TXT 支持方括号格式 `[角色] 对话` 和 JSON 行格式
- ✅ 自动拆分过长文本(可配置,默认100字符)
- ✅ 为每个角色配置自定义参考音频
- ✅ 批量合成并保留中间片段
- ✅ 可选合并为单个音频文件
- ✅ Tkinter GUI 界面
- ✅ 命令行接口

## 安装

确保已安装 qwen-tts 包:

```bash
pip install qwen-tts
```

## 使用方法

### GUI 方式

```bash
python -m scripts.batch_dialogue_tts.gui
```

### 命令行方式

```bash
# JSON 格式
python -m scripts.batch_dialogue_tts.cli \
    --input test_dialogue.json \
    --output-dir ./output \
    --merge

# TXT 格式(需先配置角色音频)
python -m scripts.batch_dialogue_tts.cli \
    --input test_dialogue.txt \
    --output-dir ./output \
    --max-chars 100
```

## 脚本格式

### JSON 格式

```json
{
  "metadata": {
    "title": "对话标题",
    "default_language": "Chinese"
  },
  "speakers": {
    "张三": {
      "ref_audio": "./ref/zhangsan.wav",
      "ref_text": "参考音频对应的文本",
      "language": "Chinese"
    }
  },
  "dialogues": [
    {"role": "张三", "text": "你好!"}
  ]
}
```

### TXT 格式 A (方括号)

```
[张三] 你好,今天天气真好!
[李四] 是啊,我们去公园吧。
```

### TXT 格式 B (JSON 行)

```
{"name": "Max_Otte", "seed": -1, "speed": 1}
黄金不再便宜了。
{"name": "Anna", "seed": 42, "speed": 1.2}
我同意你的观点。
```

## 参数说明

- `--input`: 脚本文件路径(JSON 或 TXT)
- `--output-dir`: 输出目录
- `--merge`: 是否合并为单个文件
- `--silence`: 合并时的静音间隔(毫秒)
- `--max-chars`: 文本拆分长度(默认100)
- `--model-path`: 模型路径
- `--device`: 设备(cuda:0/cpu)
