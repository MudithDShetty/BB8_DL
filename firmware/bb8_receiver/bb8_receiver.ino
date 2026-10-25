// BB-8 drive receiver for a classic ESP32 (Arduino-ESP32 core 3.x).
//
// The deep model does NOT run here. The companion app (app/companion.py) runs the
// localizer + controller and sends wheel duties over UDP at ~20 Hz. This sketch:
//   * parses command packets, drops stale/out-of-order ones,
//   * clamps duty, applies a slew-rate limit, and stops the motors if commands
//     stop arriving (watchdog) or the e-stop flag is set (latched),
//   * optionally closes a wheel-rate PI loop with quadrature encoders,
//   * sends telemetry back to whoever sent the last valid command.
//
// Packet formats (little endian), mirrored in bb8dl/protocol.py:
//   command   20 B: uint16 magic=0xB8D1, uint16 seq, uint32 t_ms, float uL, float uR, uint8 flags, 3 pad
//   telemetry 32 B: uint16 magic=0xB8D2, uint16 last_seq, uint32 t_ms, float uL_applied, float uR_applied,
//                   float wL_rad_s, float wR_rad_s, float battery_V, uint32 rx_count
//
// Motor driver assumed: TB6612FNG (two H-bridges). Pin map below.

#include <WiFi.h>
#include <WiFiUdp.h>

// ---------------- configuration -------------------------------------------
// Access-point mode by default: the laptop/phone running the app joins "BB8-DL".
// Change the passphrase before use. Do not commit real home-network credentials.
#define WIFI_AP_MODE 1
static const char *WIFI_SSID = "BB8-DL";
static const char *WIFI_PASS = "change-me-123";
static const uint16_t CMD_PORT = 5005;

#define USE_ENCODER_PI 0          // 1: interpret u as wheel-rate setpoint and run PI on encoders

static const float MAX_DUTY = 0.80f;        // hard clamp, below the cart's loop-over threshold
static const float SLEW_PER_S = 4.0f;       // duty units per second (0 -> 1 in 0.25 s)
static const uint32_t CMD_TIMEOUT_MS = 300; // watchdog
static const uint32_t CTRL_PERIOD_MS = 10;  // 100 Hz low-level loop
static const uint32_t TEL_PERIOD_MS = 100;  // 10 Hz telemetry

// TB6612FNG pins
static const int PIN_AIN1 = 25, PIN_AIN2 = 33, PIN_PWMA = 32;  // left
static const int PIN_BIN1 = 27, PIN_BIN2 = 14, PIN_PWMB = 12;  // right
static const int PIN_STBY = 13;
static const int PWM_FREQ = 20000, PWM_BITS = 10;
// Optional encoders (A/B), battery divider
static const int PIN_ENC_LA = 34, PIN_ENC_LB = 35, PIN_ENC_RA = 36, PIN_ENC_RB = 26;
static const int PIN_VBAT = 39;              // ADC1 via 1:3 divider (ADC2 cannot be read while Wi-Fi is on)
static const float ENC_TICKS_PER_RAD = 1400.0f / (2.0f * PI);
static const float W_MAX = 30.0f;            // rad/s at u = 1 (matches sim w_noload)

// ---------------- packets --------------------------------------------------
#pragma pack(push, 1)
struct CmdPacket {
  uint16_t magic;
  uint16_t seq;
  uint32_t t_ms;
  float uL;
  float uR;
  uint8_t flags;
  uint8_t pad[3];
};
struct TelPacket {
  uint16_t magic;
  uint16_t last_seq;
  uint32_t t_ms;
  float uL;
  float uR;
  float wL;
  float wR;
  float vbat;
  uint32_t rx_count;
};
#pragma pack(pop)
static_assert(sizeof(CmdPacket) == 20, "command packet must be 20 bytes");
static_assert(sizeof(TelPacket) == 32, "telemetry packet must be 32 bytes");

static const uint16_t CMD_MAGIC = 0xB8D1, TEL_MAGIC = 0xB8D2;
static const uint8_t FLAG_ESTOP = 0x01;

WiFiUDP udp;
IPAddress peerIp;
uint16_t peerPort = 0;

volatile int32_t encL = 0, encR = 0;
float targetL = 0, targetR = 0;     // from last valid packet
float appliedL = 0, appliedR = 0;   // after clamp + slew (+ PI)
float wL = 0, wR = 0;
float iL = 0, iR = 0;
uint16_t lastSeq = 0;
bool haveSeq = false;
bool estopLatched = false;
uint32_t lastCmdMs = 0, rxCount = 0;

void IRAM_ATTR isrL() { encL += (digitalRead(PIN_ENC_LB) ? -1 : 1); }
void IRAM_ATTR isrR() { encR += (digitalRead(PIN_ENC_RB) ? 1 : -1); }

static float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

static void driveOne(int in1, int in2, int pwmPin, float u) {
  u = clampf(u, -1.0f, 1.0f);
  uint32_t duty = (uint32_t)(fabsf(u) * ((1 << PWM_BITS) - 1));
  digitalWrite(in1, u > 0);
  digitalWrite(in2, u < 0);
  ledcWrite(pwmPin, duty);
}

static void drive(float uL, float uR) {
  driveOne(PIN_AIN1, PIN_AIN2, PIN_PWMA, uL);
  driveOne(PIN_BIN1, PIN_BIN2, PIN_PWMB, uR);
}

// Accept seq if it is newer than the last one (mod 2^16), or if the link has been
// silent past the watchdog (lets a restarted app, whose seq restarts at 0, take over).
static bool seqNewer(uint16_t s) {
  if (!haveSeq || millis() - lastCmdMs > CMD_TIMEOUT_MS) return true;
  return (int16_t)(s - lastSeq) > 0;
}

static void handlePackets() {
  int len;
  while ((len = udp.parsePacket()) > 0) {
    CmdPacket p;
    if (len != sizeof(CmdPacket)) { udp.flush(); continue; }
    udp.read((uint8_t *)&p, sizeof(p));
    if (p.magic != CMD_MAGIC || !isfinite(p.uL) || !isfinite(p.uR)) continue;
    if (!seqNewer(p.seq)) continue;
    haveSeq = true;
    lastSeq = p.seq;
    peerIp = udp.remoteIP();
    peerPort = udp.remotePort();
    rxCount++;
    if (p.flags & FLAG_ESTOP) {
      estopLatched = true;
    } else if (estopLatched && p.uL == 0.0f && p.uR == 0.0f) {
      estopLatched = false;  // re-arm only with an explicit zero command
    }
    targetL = clampf(p.uL, -MAX_DUTY, MAX_DUTY);
    targetR = clampf(p.uR, -MAX_DUTY, MAX_DUTY);
    lastCmdMs = millis();
  }
}

static float slew(float cur, float tgt, float dt) {
  float step = SLEW_PER_S * dt;
  return cur + clampf(tgt - cur, -step, step);
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_AIN1, OUTPUT); pinMode(PIN_AIN2, OUTPUT);
  pinMode(PIN_BIN1, OUTPUT); pinMode(PIN_BIN2, OUTPUT);
  pinMode(PIN_STBY, OUTPUT);
  digitalWrite(PIN_STBY, HIGH);
  ledcAttach(PIN_PWMA, PWM_FREQ, PWM_BITS);
  ledcAttach(PIN_PWMB, PWM_FREQ, PWM_BITS);
  drive(0, 0);
#if USE_ENCODER_PI
  pinMode(PIN_ENC_LA, INPUT); pinMode(PIN_ENC_LB, INPUT);
  pinMode(PIN_ENC_RA, INPUT); pinMode(PIN_ENC_RB, INPUT);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_LA), isrL, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_RA), isrR, RISING);
#endif
#if WIFI_AP_MODE
  WiFi.softAP(WIFI_SSID, WIFI_PASS);
  Serial.printf("AP %s  ip %s  udp %u\n", WIFI_SSID, WiFi.softAPIP().toString().c_str(), CMD_PORT);
#else
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) delay(200);
  Serial.printf("STA ip %s  udp %u\n", WiFi.localIP().toString().c_str(), CMD_PORT);
#endif
  udp.begin(CMD_PORT);
}

void loop() {
  static uint32_t lastCtrl = 0, lastTel = 0;
  static int32_t prevEncL = 0, prevEncR = 0;
  handlePackets();
  uint32_t now = millis();
  if (now - lastCtrl >= CTRL_PERIOD_MS) {
    float dt = (now - lastCtrl) * 1e-3f;
    lastCtrl = now;
    bool timedOut = (now - lastCmdMs) > CMD_TIMEOUT_MS;
    float tL = (timedOut || estopLatched) ? 0.0f : targetL;
    float tR = (timedOut || estopLatched) ? 0.0f : targetR;
    noInterrupts();
    int32_t eL = encL, eR = encR;
    interrupts();
    wL = (eL - prevEncL) / ENC_TICKS_PER_RAD / dt;
    wR = (eR - prevEncR) / ENC_TICKS_PER_RAD / dt;
    prevEncL = eL; prevEncR = eR;
#if USE_ENCODER_PI
    // u is a normalised wheel-rate setpoint; PI with anti-windup on the duty
    const float kp = 0.04f, ki = 0.6f;
    float eLr = tL * W_MAX - wL, eRr = tR * W_MAX - wR;
    iL = clampf(iL + ki * eLr * dt, -MAX_DUTY, MAX_DUTY);
    iR = clampf(iR + ki * eRr * dt, -MAX_DUTY, MAX_DUTY);
    tL = clampf(tL + kp * eLr + iL, -MAX_DUTY, MAX_DUTY);
    tR = clampf(tR + kp * eRr + iR, -MAX_DUTY, MAX_DUTY);
    if (timedOut || estopLatched) { iL = iR = 0; tL = tR = 0; }
#endif
    if (estopLatched) {
      appliedL = appliedR = 0;  // e-stop is immediate, no slew
    } else {
      appliedL = slew(appliedL, tL, dt);
      appliedR = slew(appliedR, tR, dt);
    }
    drive(appliedL, appliedR);
  }
  if (peerPort && now - lastTel >= TEL_PERIOD_MS) {
    lastTel = now;
    TelPacket t{TEL_MAGIC, lastSeq, now, appliedL, appliedR, wL, wR,
                analogReadMilliVolts(PIN_VBAT) * 3.0f / 1000.0f, rxCount};
    udp.beginPacket(peerIp, peerPort);
    udp.write((const uint8_t *)&t, sizeof(t));
    udp.endPacket();
  }
}
