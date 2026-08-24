#include <Servo.h>

Servo myServo;

int angle = 20;      // 初始角度，改这个值来设置初始状态
int minAngle = 0;
int maxAngle = 180;

void setup() {
  pinMode(2, INPUT_PULLUP);
  pinMode(3, INPUT_PULLUP);
  myServo.attach(9);
  
  // 初始化：先转到初始角度
  myServo.write(angle);
  delay(1000);  // 等待舵机到位
  
  Serial.begin(9600);
  Serial.print("初始角度: ");
  Serial.println(angle);
}

void loop() {

}
