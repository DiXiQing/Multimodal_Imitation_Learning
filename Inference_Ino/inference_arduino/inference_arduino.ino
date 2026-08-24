#include <Servo.h>

Servo myServo;

int angle     = 10;
int minAngle  = 0;
int maxAngle  = 70;

void setup() {
  myServo.attach(9);
  myServo.write(angle);
  Serial.begin(9600);
}

void loop() {
  // 等待 Python 发来的指令，格式: "CMD:45\n"
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.startsWith("CMD:")) {
      int newAngle = line.substring(4).toInt();

      // 限幅保护
      newAngle = constrain(newAngle, minAngle, maxAngle);

      if (newAngle != angle) {
        angle = newAngle;
        myServo.write(angle);
      }
    }
  }
}
