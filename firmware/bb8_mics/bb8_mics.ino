// BB-8 4-mic capture for a classic ESP32 (Arduino-ESP32 core 3.x, ESP-IDF 5 I2S std driver).
//
// Four INMP441 on two I2S buses, raw 48 kHz streaming to the companion app over UDP.
// No localization runs here: the deep model needs a GPU/CPU-class host. See
// firmware/README.md for the wiring and the capture plan.
//
//   I2S0 = master: drives BCLK (3.072 MHz = 64 x 48 kHz) and WS; data from mics 0 (L) and 1 (R)
//   I2S1 = slave:  BCLK/WS inputs jumpered from the I2S0 pins; data from mics 2 (L) and 3 (R)
// Sharing one clock keeps all four channels sample-synchronous. The two buses can still
// start a whole frame apart; calibrate that once (README, "bus offset").
//
// Packet (bb8dl/protocol.py): uint16 magic 0xB8A1, uint16 seq, uint32 first_sample, then
// 180 frames x 4 x int16 (1448 B). 48 kHz x 4 ch x 16 bit = 3.07 Mbit/s of payload.
// Streaming starts when an app sends "SUB1" to UDP 5006 and stops 3 s after the last one.

#include <WiFi.h>
#include <WiFiUdp.h>
#include "driver/i2s_std.h"

// ---------------- configuration -------------------------------------------
static const char *WIFI_SSID = "BB8-DL";          // join the drive ESP32's AP, or a hotspot/router
static const char *WIFI_PASS = "change-me-123";   // placeholder; never commit real credentials
static const IPAddress MY_IP(192, 168, 4, 3), GATEWAY(192, 168, 4, 1), SUBNET(255, 255, 255, 0);
static const uint16_t MIC_PORT = 5006;
static const uint32_t FS = 48000;
static const int FRAMES = 180;
static const int SHIFT = 14;                        // 24-bit sample in a 32-bit slot -> int16 (+4x gain over >>16)
static const uint32_t SUB_TIMEOUT_MS = 3000;

static const int PIN_BCLK0 = 26, PIN_WS0 = 25, PIN_SD0 = 33;   // master outputs + data in
static const int PIN_BCLK1 = 14, PIN_WS1 = 27, PIN_SD1 = 32;   // slave clock inputs (jumper 26->14, 25->27)

// ---------------- state -----------------------------------------------------
static i2s_chan_handle_t rx0, rx1;
static int32_t raw0[FRAMES * 2], raw1[FRAMES * 2];
#pragma pack(push, 1)
struct AudioPacket {
  uint16_t magic;
  uint16_t seq;
  uint32_t first;
  int16_t s[FRAMES * 4];
};
#pragma pack(pop)
static_assert(sizeof(AudioPacket) == 8 + FRAMES * 8, "audio packet layout");
static AudioPacket pkt;
static WiFiUDP udp;
static IPAddress peerIp;
static uint16_t peerPort = 0;
static uint32_t lastSubMs = 0, sampleIndex = 0, overruns = 0;
static uint16_t seq = 0;

static inline int16_t to16(int32_t v) {
  int32_t x = v >> SHIFT;
  return (int16_t)(x > 32767 ? 32767 : (x < -32768 ? -32768 : x));
}

static void initI2S() {
  i2s_chan_config_t c0 = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  i2s_chan_config_t c1 = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_1, I2S_ROLE_SLAVE);
  c0.dma_desc_num = c1.dma_desc_num = 8;
  c0.dma_frame_num = c1.dma_frame_num = FRAMES;
  ESP_ERROR_CHECK(i2s_new_channel(&c0, NULL, &rx0));
  ESP_ERROR_CHECK(i2s_new_channel(&c1, NULL, &rx1));

  i2s_std_config_t s0 = {
      .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(FS),
      .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
      .gpio_cfg = {.mclk = I2S_GPIO_UNUSED, .bclk = (gpio_num_t)PIN_BCLK0, .ws = (gpio_num_t)PIN_WS0,
                   .dout = I2S_GPIO_UNUSED, .din = (gpio_num_t)PIN_SD0,
                   .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false}},
  };
  i2s_std_config_t s1 = s0;
  s1.gpio_cfg.bclk = (gpio_num_t)PIN_BCLK1;
  s1.gpio_cfg.ws = (gpio_num_t)PIN_WS1;
  s1.gpio_cfg.din = (gpio_num_t)PIN_SD1;
  ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx0, &s0));
  ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx1, &s1));
  // slave first, so it is already waiting when the master starts the clocks
  ESP_ERROR_CHECK(i2s_channel_enable(rx1));
  ESP_ERROR_CHECK(i2s_channel_enable(rx0));
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  WiFi.config(MY_IP, GATEWAY, SUBNET);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) delay(200);
  WiFi.setSleep(false);  // modem sleep adds 100 ms-scale latency spikes
  Serial.printf("STA ip %s  udp %u\n", WiFi.localIP().toString().c_str(), MIC_PORT);
  udp.begin(MIC_PORT);
  initI2S();
  pkt.magic = 0xB8A1;
}

void loop() {
  size_t got0 = 0, got1 = 0;
  // blocking reads pace the loop at 48 kHz / 180 = 267 Hz
  i2s_channel_read(rx0, raw0, sizeof(raw0), &got0, portMAX_DELAY);
  i2s_channel_read(rx1, raw1, sizeof(raw1), &got1, portMAX_DELAY);
  if (got0 != sizeof(raw0) || got1 != sizeof(raw1)) overruns++;

  int len;
  while ((len = udp.parsePacket()) > 0) {
    char b[8] = {0};
    udp.read(b, sizeof(b) - 1);
    if (len == 4 && memcmp(b, "SUB1", 4) == 0) {
      peerIp = udp.remoteIP();
      peerPort = udp.remotePort();
      lastSubMs = millis();
    }
  }

  if (peerPort && millis() - lastSubMs < SUB_TIMEOUT_MS) {
    for (int i = 0; i < FRAMES; i++) {
      pkt.s[4 * i + 0] = to16(raw0[2 * i]);      // mic 0: bus 0, L/R pin low
      pkt.s[4 * i + 1] = to16(raw0[2 * i + 1]);  // mic 1: bus 0, L/R pin high
      pkt.s[4 * i + 2] = to16(raw1[2 * i]);      // mic 2: bus 1, L/R pin low
      pkt.s[4 * i + 3] = to16(raw1[2 * i + 1]);  // mic 3: bus 1, L/R pin high
    }
    pkt.seq = seq++;
    pkt.first = sampleIndex;
    udp.beginPacket(peerIp, peerPort);
    udp.write((const uint8_t *)&pkt, sizeof(pkt));
    udp.endPacket();
  }
  sampleIndex += FRAMES;
  static uint32_t lastLog = 0;
  if (millis() - lastLog > 5000) {
    lastLog = millis();
    Serial.printf("samples %lu  short reads %lu  subscribed %d\n", (unsigned long)sampleIndex,
                  (unsigned long)overruns, peerPort && millis() - lastSubMs < SUB_TIMEOUT_MS);
  }
}
