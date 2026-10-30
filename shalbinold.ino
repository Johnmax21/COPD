/*
  esp32_combined_breath_and_dose.ino
  ====================================
  Combines two subsystems on one ESP32, sharing one WiFi connection:

  1. BREATH VOC MONITOR (MQ135 pin 32, MQ137 pin 33)
     - Press ENTER in Serial Monitor to trigger a 5-second breath measurement
     - Sends the breath's readings to Django -> /predict/

  2. INHALER DOSE TRACKER (pressure sensor pin 34)
     - Continuously monitors for cartridge presses
     - Each detected dose is sent immediately to Django -> /record-dose/

  NOTE ON TIMING: while a breath measurement is running (5 seconds, with a
  delay(100) between samples), dose detection is paused -- the ESP32 can't
  do both at the exact same instant with this simple structure. In practice
  this is a 5-second window every time you take a breath test, which is a
  minor limitation worth mentioning in your report rather than a bug.
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ---- CONFIG: fill these in --------------------------------------------
const char* WIFI_SSID      = "Galaxy";
const char* WIFI_PASSWORD  = "88888888";
const char* PREDICT_URL    = "http://10.141.94.6:8000/predict/";      // <-- update IP if it changes
const char* DOSE_URL       = "http://10.141.94.6:8000/record-dose/";  // <-- update IP if it changes
// -------------------------------------------------------------------------

// ---- Breath monitor pins/config ----
#define MQ135_PIN 32
#define MQ137_PIN 33
const unsigned long BREATH_TIME = 5000;
const unsigned long SAMPLE_INTERVAL = 100;

int mq135Baseline = 0;
int mq137Baseline = 0;
int mq135Max = 0;
int mq137Max = 0;

const int MAX_SAMPLES = 100;
int mq135Readings[MAX_SAMPLES];
int mq137Readings[MAX_SAMPLES];
int sampleCount = 0;

bool measuring = false;
unsigned long breathStart = 0;

// ---- Dose tracker pins/config ----
#define PRESSURE_SENSOR_PIN 34
int PRESSURE_THRESHOLD = 1300;
int doseCount = 0;
bool doseDetected = false;
unsigned long lastDoseTime = 0;


void connectWiFi() {
  Serial.print("Connecting to WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected. IP address: " + WiFi.localIP().toString());
}


void sendBreathDataToServer() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected -- skipping send.");
    return;
  }

  StaticJsonDocument<4096> doc;
  JsonArray mq135_arr = doc.createNestedArray("mq135");
  JsonArray mq137_arr = doc.createNestedArray("mq137");

  for (int i = 0; i < sampleCount; i++) {
    mq135_arr.add(mq135Readings[i]);
    mq137_arr.add(mq137Readings[i]);
  }

  String payload;
  serializeJson(doc, payload);

  HTTPClient http;
  http.begin(PREDICT_URL);
  http.addHeader("Content-Type", "application/json");

  Serial.println("Sending breath data to server...");
  int httpCode = http.POST(payload);

  if (httpCode > 0) {
    Serial.println("Server response: " + http.getString());
  } else {
    Serial.println("HTTP POST failed, error: " + http.errorToString(httpCode));
  }
  http.end();
}


void sendDoseEventToServer(int doseNumber, unsigned long intervalSeconds) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected -- skipping dose send.");
    return;
  }

  StaticJsonDocument<256> doc;
  doc["dose_number"] = doseNumber;
  doc["interval_seconds"] = intervalSeconds;

  String payload;
  serializeJson(doc, payload);

  HTTPClient http;
  http.begin(DOSE_URL);
  http.addHeader("Content-Type", "application/json");

  int httpCode = http.POST(payload);
  if (httpCode > 0) {
    Serial.println("Dose logged. Server response: " + http.getString());
  } else {
    Serial.println("Dose POST failed, error: " + http.errorToString(httpCode));
  }
  http.end();
}


void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  pinMode(PRESSURE_SENSOR_PIN, INPUT);

  Serial.println();
  Serial.println("================================");
  Serial.println("MQ135 + MQ137 BREATH MONITOR + DOSE TRACKER");
  Serial.println("================================");

  connectWiFi();

  Serial.println("Stabilizing sensors...");
  delay(10000);

  long total135 = 0;
  for (int i = 0; i < 50; i++) {
    total135 += analogRead(MQ135_PIN);
    delay(50);
  }
  mq135Baseline = total135 / 50;

  long total137 = 0;
  for (int i = 0; i < 50; i++) {
    total137 += analogRead(MQ137_PIN);
    delay(50);
  }
  mq137Baseline = total137 / 50;

  Serial.println();
  Serial.println("--------------------------------");
  Serial.print("MQ135 Baseline = ");
  Serial.println(mq135Baseline);
  Serial.print("MQ137 Baseline = ");
  Serial.println(mq137Baseline);
  Serial.println("--------------------------------");
  Serial.println("Press ENTER to start exhaled breath measurement.");
  Serial.println("(Dose tracking runs continuously in the background.)");
  Serial.println();
}


void loop() {

  // ==================================================
  // DOSE TRACKING (runs continuously unless a breath measurement is active)
  // ==================================================
  if (!measuring) {
    int pressureValue = analogRead(PRESSURE_SENSOR_PIN);

    if (pressureValue >= PRESSURE_THRESHOLD) {
      if (doseDetected == false) {
        doseDetected = true;
        doseCount++;

        unsigned long currentTime = millis();
        unsigned long intervalSeconds = 0;

        if (doseCount > 1) {
          intervalSeconds = (currentTime - lastDoseTime) / 1000;
        }

        Serial.println();
        Serial.print("DOSE ");
        Serial.println(doseCount);
        Serial.print("Interval: ");
        Serial.print(intervalSeconds);
        Serial.println(" sec");

        sendDoseEventToServer(doseCount, intervalSeconds);

        lastDoseTime = currentTime;
      }
    } else {
      doseDetected = false;
    }
  }


  // ==================================================
  // WAIT FOR ENTER -> START BREATH MEASUREMENT
  // ==================================================
  if (!measuring && Serial.available() > 0) {
    while (Serial.available() > 0) {
      Serial.read();
    }

    measuring = true;
    breathStart = millis();
    sampleCount = 0;

    mq135Max = analogRead(MQ135_PIN);
    mq137Max = analogRead(MQ137_PIN);

    Serial.println();
    Serial.println("================================");
    Serial.println("EXHALE NOW...");
    Serial.println("================================");
  }


  // ==================================================
  // CONTINUOUS BREATH MEASUREMENT
  // ==================================================
  if (measuring) {
    int mq135 = analogRead(MQ135_PIN);
    int mq137 = analogRead(MQ137_PIN);

    if (mq135 > mq135Max) mq135Max = mq135;
    if (mq137 > mq137Max) mq137Max = mq137;

    Serial.print("MQ135: ");
    Serial.print(mq135);
    Serial.print(" | MQ137: ");
    Serial.println(mq137);

    if (sampleCount < MAX_SAMPLES) {
      mq135Readings[sampleCount] = mq135;
      mq137Readings[sampleCount] = mq137;
      sampleCount++;
    }

    if (millis() - breathStart >= BREATH_TIME) {
      measuring = false;

      Serial.println();
      Serial.println("--------------------------------");
      Serial.println("BREATH MEASUREMENT COMPLETE");
      Serial.println("--------------------------------");
      Serial.print("MQ135 Baseline = ");
      Serial.println(mq135Baseline);
      Serial.print("MAX MQ135 = ");
      Serial.println(mq135Max);
      Serial.print("MQ135 Change = ");
      Serial.println(mq135Max - mq135Baseline);
      Serial.println();
      Serial.print("MQ137 Baseline = ");
      Serial.println(mq137Baseline);
      Serial.print("MAX MQ137 = ");
      Serial.println(mq137Max);
      Serial.print("MQ137 Change = ");
      Serial.println(mq137Max - mq137Baseline);
      Serial.println("--------------------------------");

      sendBreathDataToServer();

      Serial.println();
      Serial.println("Press ENTER for next breath.");
      Serial.println();
    }
  }

  delay(50);
}
