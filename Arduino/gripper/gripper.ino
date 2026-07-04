#include <Servo.h>

Servo myServo;

int angle    = 10;
int minAngle = 0;
int maxAngle = 70;

void setup() {
  pinMode(2, INPUT_PULLUP);  // 夹紧按钮
  pinMode(3, INPUT_PULLUP);  // 张开按钮
  myServo.attach(9);
  myServo.write(angle);
  Serial.begin(9600);
}

void loop() {

  // ── 串口指令接收 ──
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.startsWith("CMD:")) {
      // 推理模式：直接写角度
      // 格式: "CMD:45"
      int newAngle = line.substring(4).toInt();
      angle = constrain(newAngle, minAngle, maxAngle);
      myServo.write(angle);

    } else if (line == "O") {
      // 录制模式：张开一步
      if (angle < maxAngle) {
        angle++;
        myServo.write(angle);
      }

    } else if (line == "C") {
      // 录制模式：夹紧一步
      if (angle > minAngle) {
        angle--;
        myServo.write(angle);
      }
    }

    // 执行完指令后回报当前角度
    Serial.print("ANGLE:");
    Serial.println(angle);
  }

  // ── 物理按钮控制（录制时备用）──
  if (digitalRead(2) == LOW) {
    if (angle > minAngle) {
      angle--;
      myServo.write(angle);
      Serial.print("ANGLE:");
      Serial.println(angle);
    }
  }

  if (digitalRead(3) == LOW) {
    if (angle < maxAngle) {
      angle++;
      myServo.write(angle);
      Serial.print("ANGLE:");
      Serial.println(angle);
    }
  }

  delay(20);
}
