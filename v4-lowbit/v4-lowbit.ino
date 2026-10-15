#include "esp_partition.h"
#include "esp_heap_caps.h"
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C

#define SDA_PIN 23
#define SCL_PIN 21

#define CHARS_PER_LINE 21  // 128 px / 6 px per char at text size 1
#define TEXT_LINES 7       // bottom line is kept for the speed
#define STORY_START "Once upon a time"
#define TEMPERATURE 0.8f
#define PAUSE_MS 5000
#define BENCH_RUNS 3

// Raw files written into the data partition (0x290000) with esptool:
//   stories260K_bN.bin (from tools/quantize_bits.py) at offset 0, tok512.bin at TOKENIZER_OFFSET (flash 0x3A0000).
#define TOKENIZER_OFFSET 0x110000

extern "C" {
  int run_llama(const uint8_t* model, size_t model_size, const uint8_t* tokenizer_data, size_t tokenizer_size,
                float temperature, float topp, int steps, const char* prompt, int weights_in_ram);
  extern int llama_ram_percent;
  extern int llama_bits;
}

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
QueueHandle_t promptQueue;

String story;
int tokens = 0;
unsigned long firstTokenMs = 0, lastTokenMs = 0;

float tokensPerSecond() {
  return tokens > 1 ? (tokens - 1) * 1000.0 / (lastTokenMs - firstTokenMs) : 0;
}

// Word-wrap the story and show its last lines, so the text scrolls up as it grows.
void render() {
  String lines[64];
  int count = 0;
  String line, word;
  for (int i = 0; i <= (int)story.length() && count < 63; i++) {
    char c = i < (int)story.length() ? story[i] : ' ';
    if (c != ' ' && c != '\n') {
      word += c;
      continue;
    }
    while (word.length() > CHARS_PER_LINE) {  // a word wider than the screen gets split
      lines[count++] = word.substring(0, CHARS_PER_LINE);
      word = word.substring(CHARS_PER_LINE);
    }
    if (line.length() + (line.length() ? 1 : 0) + word.length() > CHARS_PER_LINE) {
      lines[count++] = line;
      line = "";
    }
    if (word.length()) line += (line.length() ? " " : "") + word;
    word = "";
    if (c == '\n' && line.length()) {
      lines[count++] = line;
      line = "";
    }
  }
  if (line.length()) lines[count++] = line;

  display.clearDisplay();
  int first = max(0, count - TEXT_LINES);
  for (int i = first; i < count; i++) {
    display.setCursor(0, (i - first) * 8);
    display.print(lines[i]);
  }
  if (tokens > 1) {
    display.setCursor(0, 56);
    display.printf("%db %d%%RAM %.1f tok/s", llama_bits, llama_ram_percent, tokensPerSecond());
  }
  display.display();
}

extern "C" void on_token(const char *piece) {
  lastTokenMs = millis();
  if (tokens++ == 0) firstTokenMs = lastTokenMs;
  story += piece;
  render();
}

void showMessage(const char *msg) {
  display.clearDisplay();
  display.setCursor(0, 28);
  display.print(msg);
  display.display();
}

const esp_partition_t *part;
const uint8_t *flash;

// One story; returns false if the model or tokenizer in flash is invalid.
bool tellStory(const char *prompt, float temperature, int inRam) {
  story = "";
  tokens = 0;
  showMessage("Thinking...");
  return run_llama(flash, TOKENIZER_OFFSET, flash + TOKENIZER_OFFSET, part->size - TOKENIZER_OFFSET,
                   temperature, 0.9f, 250, prompt, inRam);
}

// Greedy story from an empty prompt (compare with the laptop), then speed with weights in flash and in RAM.
bool benchmark() {
  Serial.printf("BENCH heap free %u, largest block %u\n",
                heap_caps_get_free_size(MALLOC_CAP_8BIT), heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
  if (!tellStory("", 0.0f, 0)) return false;
  Serial.printf("BENCH bits=%d greedy story: %s\n", llama_bits, story.c_str());
  for (int inRam = 0; inRam <= 1; inRam++) {
    float sum = 0;
    for (int r = 0; r < BENCH_RUNS; r++) {
      tellStory(STORY_START, 0.0f, inRam);
      sum += tokensPerSecond();
    }
    Serial.printf("BENCH bits=%d mode=%s weights_in_ram=%d%% tokens=%d tok/s=%.2f\n", llama_bits,
                  inRam ? "RAM" : "flash", llama_ram_percent, tokens, sum / BENCH_RUNS);
  }
  return true;
}

void llamaTask(void *pvParameters) {
  // Map the whole data partition so weights are read through the flash cache, not a filesystem.
  part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_SPIFFS, NULL);
  const void *base = NULL;
  esp_partition_mmap_handle_t mmapHandle;
  if (!part || part->size <= TOKENIZER_OFFSET ||
      esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &base, &mmapHandle) != ESP_OK) {
    Serial.println("Could not map the model partition");
    showMessage("Model not in flash");
    vTaskDelete(NULL);
    return;
  }
  flash = (const uint8_t *)base;

  if (!benchmark()) {
    Serial.println("Model or tokenizer in flash is invalid: write the raw .bin files (see top of sketch)");
    showMessage("Model not in flash");
    vTaskDelete(NULL);
    return;
  }

  char prompt[256];
  for (;;) {
    // A prompt typed in the Serial Monitor replaces the default for one story.
    if (xQueueReceive(promptQueue, &prompt, 0) != pdTRUE) strcpy(prompt, STORY_START);
    Serial.printf("\nPrompt: \"%s\"\n", prompt);
    tellStory(prompt, TEMPERATURE, 1);  // as much of the weights in RAM as fits
    vTaskDelay(pdMS_TO_TICKS(PAUSE_MS));
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);

  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println("SSD1306 not found");
    while (true);
  }
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  showMessage("Loading model...");

  promptQueue = xQueueCreate(5, sizeof(char[256]));
  xTaskCreatePinnedToCore(llamaTask, "LlamaTask", 24576, NULL, 1, NULL, 1);
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      char prompt[256];
      input.toCharArray(prompt, sizeof(prompt));
      xQueueSend(promptQueue, &prompt, pdMS_TO_TICKS(100));
    }
  }
  vTaskDelay(pdMS_TO_TICKS(50));
}
