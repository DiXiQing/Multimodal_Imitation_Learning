#include <Servo.h>

Servo myServo;

// ===== 改这一个数就行 (0-70度) =====
int targetAngle = 30;
// ==================================

const int SERVO_PIN = 9;

void setup() {
  myServo.attach(SERVO_PIN);
  myServo.write(targetAngle);
}

void loop() {
  // 什么都不用做，舵机会保持在 targetAngle
}
