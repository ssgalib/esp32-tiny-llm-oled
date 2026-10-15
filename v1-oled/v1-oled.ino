#include "FS.h"
#include "LittleFS.h"
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
#define PAUSE_MS 5000

extern "C" {
  void run_llama(const char* model_path, const char* tokenizer_path, float temperature, float topp, int steps, const char* prompt);
}

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
QueueHandle_t promptQueue;

String story;
int tokens = 0;
unsigned long firstTokenMs = 0;

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
    display.printf("%.1f tok/s", (tokens - 1) * 1000.0 / (millis() - firstTokenMs));
  }
  display.display();
}

extern "C" void on_token(const char *piece) {
  if (tokens++ == 0) firstTokenMs = millis();
  story += piece;
  render();
}

void showMessage(const char *msg) {
  display.clearDisplay();
  display.setCursor(0, 28);
  display.print(msg);
  display.display();
}

void llamaTask(void *pvParameters) {
  if (!LittleFS.begin(false) || !LittleFS.exists("/stories260K.bin") || !LittleFS.exists("/tok512.bin")) {
    Serial.println("Model files missing: upload littlefs.bin to 0x290000");
    showMessage("Model files missing");
    vTaskDelete(NULL);
    return;
  }

  char prompt[256];
  for (;;) {
    // A prompt typed in the Serial Monitor replaces the default for one story.
    if (xQueueReceive(promptQueue, &prompt, 0) != pdTRUE) strcpy(prompt, STORY_START);

    story = "";
    tokens = 0;
    showMessage("Thinking...");
    Serial.printf("\nPrompt: \"%s\"\n", prompt);
    run_llama("/stories260K.bin", "/tok512.bin", 0.8f, 0.9f, 250, prompt);

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
