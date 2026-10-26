// BB-8 acoustic beacon, stage 1: two INMP441 mics on one I2S port.
// Arduino IDE, ESP32 board package 3.x (uses the new I2S driver).
//
// Serial commands (921600 baud):
//   'm' -> prints one line with the level of each mic in dBFS
//   'r' -> records CAPTURE_MS of stereo audio and sends it as binary
//          ("AUD0", uint32 frames, uint32 sample rate, int16 L,R,L,R...)
//
// Wiring (both mics share the same three signal wires):
//   INMP441   ESP32 classic   ESP32-S3
//   VDD       3V3             3V3
//   GND       GND             GND
//   SCK       GPIO 26         GPIO 4
//   WS        GPIO 25         GPIO 5
//   SD        GPIO 33         GPIO 6
//   L/R       mic A -> GND (LEFT),  mic B -> 3V3 (RIGHT)

#include <Arduino.h>
#include "driver/i2s_std.h"

#if CONFIG_IDF_TARGET_ESP32S3
  #define PIN_SCK 4
  #define PIN_WS  5
  #define PIN_SD  6
#else  // classic ESP32 / ESP32-C3: change if these pins are taken on your board
  #define PIN_SCK 26
  #define PIN_WS  25
  #define PIN_SD  33
#endif

#define SAMPLE_RATE 48000
#define CAPTURE_MS  300
#define SHIFT       14     // 32-bit I2S word -> int16. Lower = more gain (clips sooner).

static i2s_chan_handle_t rx;
static int32_t raw[512];   // 256 stereo frames per read
static int16_t *rec;
static const size_t N = (size_t)SAMPLE_RATE * CAPTURE_MS / 1000;

static inline int16_t sat16(int32_t v) {
  return v > 32767 ? 32767 : (v < -32768 ? -32768 : (int16_t)v);
}

static void setupI2S() {
  i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  cc.dma_desc_num = 8;
  cc.dma_frame_num = 256;
  ESP_ERROR_CHECK(i2s_new_channel(&cc, NULL, &rx));

  i2s_std_config_t sc = {
    .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
    .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
    .gpio_cfg = {
      .mclk = I2S_GPIO_UNUSED,
      .bclk = (gpio_num_t)PIN_SCK,
      .ws   = (gpio_num_t)PIN_WS,
      .dout = I2S_GPIO_UNUSED,
      .din  = (gpio_num_t)PIN_SD,
      .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
    },
  };
  ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx, &sc));
  ESP_ERROR_CHECK(i2s_channel_enable(rx));
}

static size_t readFrames() {           // returns stereo frames read into raw[]
  size_t bytes = 0;
  i2s_channel_read(rx, raw, sizeof(raw), &bytes, portMAX_DELAY);
  return bytes / 8;
}

static void flush() { for (int i = 0; i < 10; i++) readFrames(); }

static void meter() {
  flush();
  double sL = 0, sR = 0; size_t n = 0;
  while (n < SAMPLE_RATE / 5) {        // 200 ms
    size_t f = readFrames();
    for (size_t i = 0; i < f; i++) {
      double l = (raw[2*i] >> 8), r = (raw[2*i+1] >> 8);   // 24-bit samples
      sL += l * l; sR += r * r;
    }
    n += f;
  }
  const double fs = 8388608.0;         // 2^23
  double dL = 20 * log10(sqrt(sL / n) / fs + 1e-12);
  double dR = 20 * log10(sqrt(sR / n) / fs + 1e-12);
  Serial.printf("L %6.1f dBFS   R %6.1f dBFS%s\n", dL, dR,
                (dL < -110 || dR < -110) ? "   <- a mic looks dead/unwired" : "");
}

static void capture() {
  flush();
  size_t got = 0;
  while (got < N) {
    size_t f = readFrames();
    for (size_t i = 0; i < f && got < N; i++, got++) {
      rec[2*got]   = sat16(raw[2*i]   >> SHIFT);
      rec[2*got+1] = sat16(raw[2*i+1] >> SHIFT);
    }
  }
  uint32_t n = N, fs = SAMPLE_RATE;
  Serial.write((const uint8_t*)"AUD0", 4);
  Serial.write((const uint8_t*)&n, 4);
  Serial.write((const uint8_t*)&fs, 4);
  Serial.write((const uint8_t*)rec, N * 4);
  Serial.flush();
}

void setup() {
  Serial.begin(921600);
  rec = (int16_t*)malloc(N * 2 * sizeof(int16_t));
  if (!rec) { Serial.println("malloc failed, lower CAPTURE_MS"); while (1) delay(1000); }
  setupI2S();
  delay(200);   // INMP441 needs ~85 ms to start up
  flush();
  Serial.println("ready: send 'm' for levels, 'r' to record");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'm') meter();
    else if (c == 'r') capture();
  }
}
