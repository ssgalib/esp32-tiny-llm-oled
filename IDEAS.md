# Tiny LLM on ESP32: ideas

Model: `stories260K` (260K params, 1 MB, TinyStories). It continues a story opening; it does not chat or follow instructions.
Board: ESP32-D0WD-V3, no PSRAM, 4 MB flash, 128×64 SSD1306 OLED on SDA 23 / SCL 21.
Baseline from the upstream repo: 2.3 tok/s on a non-PSRAM ESP32 (measured 2.1 tok/s on this board with LittleFS; 10.1 tok/s after #4; 15.3 tok/s with the int8 model).

| # | Idea | What it involves | Difficulty | Status |
|---|------|------------------|------------|--------|
| 1 | Story feed on the OLED | Story appears word by word with tok/s; a new story starts when one ends | Easy | Done: `v1-oled/`, 2.1 tok/s, stories cut at 64 tokens |
| 2 | Story prompts from a phone | ESP32 serves a web page; typed opening plays on the OLED. WiFi competes with the model for RAM | Medium | |
| 3 | Story clock | Combine with `esp32-codes/wifi_clock`: a new story every hour, opening with the time | Easy–medium | |
| 4 | Read the model straight from flash | Memory-map the flash partition instead of `fread` through LittleFS. Likely 2–4× faster | Medium | Done: `v2-mmap/`, 10.1 tok/s (4.8×). Raw model at 0x290000, tokenizer at 0x3A0000 |
| 5 | 8-bit model in internal RAM | Quantize 1 MB → ~276 KB, keep it in SRAM, split matmuls across both cores. Whole model + KV cache (~360 KB) does not fit in ~300 KB free RAM, so only parts can move to RAM | Hard | Partly done: `v3-q8/`, int8 model read from flash, 15.3 tok/s (`tools/quantize.py`). Not done: RAM placement, dual core |
| 6 | Train your own tiny model | `llama2.c` training on the laptop (fortunes, haiku, Bangla text…), then flash it | Medium–hard | |

Suggested order: 1, then 5.
