#include <Servo.h>

Servo myServo;

int angle = 10;      // 初始角度，从中间开始
int minAngle = 0;    // 最大夹紧
int maxAngle = 70;  // 最大张开

void setup() {
  pinMode(2, INPUT_PULLUP);
  pinMode(3, INPUT_PULLUP);
  myServo.attach(9);
  myServo.write(angle);
  Serial.begin(9600);
}

void loop() {
  // 接收Python发送的控制命令
  if (Serial.available()) {
    char cmd = Serial.read();

    if (cmd == 'O') {       // Open
      if (angle < maxAngle) {
        angle++;
        myServo.write(angle);
      }
    }

    if (cmd == 'C') {       // Close
      if (angle > minAngle) {
        angle--;
        myServo.write(angle);
      }
    }
  }

  if (digitalRead(2) == LOW) {  // 按住慢慢夹紧
    if (angle > minAngle) {
      angle--;
      myServo.write(angle);

    }
  }
  if (digitalRead(3) == LOW) {  // 按住慢慢张开
    if (angle < maxAngle) {
      angle++;
      myServo.write(angle);
    }
  }
  delay(20);  // 这个值控制速度，越小越快
}