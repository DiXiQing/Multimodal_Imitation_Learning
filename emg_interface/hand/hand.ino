#include <Pixy2.h>
#include <PIDLoop.h>

Pixy2 pixy;
PIDLoop panLoop(170, 0, 30, true);

char var;
int color_choice;  //色のコード指定0b0000000
int32_t panOffset, tiltOffset;

const int SV_PIN = 6;  // 舵机控制引脚
int32_t grip = 250,    // 抓取舵机初始值
        pan = 700,  //実験用右向き
        tilt = 800;    // 垂直舵机初始值

void setup(){
  Serial.begin(9600);
  pixy.init();
  analogWrite(SV_PIN, grip);
  pixy.setServos(pan, tilt);
}

void loop(){
  if (Serial.available() > 0){  // シリアルポートで受信したデータが存在する,这一步是必要的,否则会卡死
    var = Serial.read();  //動作結果

    switch (var) {  //ボタンを押したら閉じる・開く，10度ずつ変更
      case '1':  //閉じる
        if (grip < 215) {
          grip += 20;
          analogWrite(SV_PIN, grip);
        }
        break;

      case '2':  //開く
        if (grip > 130) {
          grip -= 20;
        } else if (grip > 110){
          grip -= 5;
        }
        analogWrite(SV_PIN, grip);
        break;

      case '3':  // Move left
        pan += 50;
        pixy.setServos(pan, tilt);
        //Serial.print("Moving left to pan: ");
        //Serial.println(pan);  // Debugging statement
        break;

      case '4':  // Move right
        pan -= 50;
        pixy.setServos(pan, tilt);
        //Serial.print("Moving right to pan: ");
        //Serial.println(pan);  // Debugging statement
        break;
    }
  }
}
