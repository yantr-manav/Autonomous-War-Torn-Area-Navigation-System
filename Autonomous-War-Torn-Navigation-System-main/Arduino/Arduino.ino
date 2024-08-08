//Libraries to import
#include <stdbool.h>
#include <L298NX2.h>
#include <Arduino.h>
#include <string.h>
#include <BluetoothSerial.h>
#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run make menuconfig to and enable it
#endif
#include <string.h>

#define farleft 23
#define left 22
#define center 19
#define right 18
#define farright 15


// Defining motors:
#define PWMA 13 // right
#define motorr1 12
#define motorr2 14
#define PWMB 25 // left
#define motorl1 26
#define motorl2 27

//Declaring objects
BluetoothSerial SerialBT;
L298NX2 motors(PWMA, motorl1, motorl2, PWMB, motorr1, motorr2); // library

// Defining buzzer and led:
#define buzzer 33
#define led 32

// IR VARIABLES
int leftfarR;
int leftR;
int centerR;
int rightR;
int rightfarR;

// OTHERS
int counter = -1;
int flag = 0;
int FlagX = 0;
int flagX = 0;
int loop_counter = 0;

String queue;
char charArray[100];

int array_counter_event = 0;
char array_element_event_checker = 'a';
// int strLength = str.length();

// Defining functions:
void forward();
void backward();
void stop();
void leftturn();
void rightturn();
void node();
void readsensors();
void detectevent();
void inch();
void linch();
void rinch();

void setup()
{
    //Bluetooth setup
    Serial.begin(9600);
    SerialBT.begin("ESP32_BT"); 
    Serial.println("The device started, now you can pair it with bluetooth!");

    // IR SENSORS
    pinMode(farleft, INPUT);
    pinMode(left, INPUT);
    pinMode(farright, INPUT);
    pinMode(right, INPUT);
    pinMode(center, INPUT);

    // MOTORS
    pinMode(motorl1, OUTPUT);
    pinMode(motorl2, OUTPUT);
    pinMode(motorr1, OUTPUT);
    pinMode(motorr2, OUTPUT);
    pinMode(PWMA, OUTPUT);
    pinMode(PWMB, OUTPUT);

    // OTHERS
    pinMode(buzzer, OUTPUT);
    pinMode(led, OUTPUT);

}

void readsensors()
{
    leftfarR = digitalRead(farleft);
    centerR = digitalRead(center);
    leftR = digitalRead(left);
    rightR = digitalRead(right);
    rightfarR = digitalRead(farright);
}

void forward()
{
    digitalWrite(motorr1, HIGH);
    digitalWrite(motorr2, LOW);
    digitalWrite(motorl1, HIGH);
    digitalWrite(motorl2, LOW);
    analogWrite(PWMA, 80);
    analogWrite(PWMB, 80);
}

void rightturn()
{
    digitalWrite(motorl1, HIGH);
    digitalWrite(motorl2, LOW);
    digitalWrite(motorr1, LOW);
    digitalWrite(motorr2, LOW);
    analogWrite(PWMA, 220);
    analogWrite(PWMB, 50);
}

void backward()
{
  motors.backward();
  motors.setSpeedA(200);
  motors.setSpeedB(200);
  delay(750);
  motors.backward();
  motors.setSpeedA(0);
  motors.setSpeedB(0);
  delay(500);
}

void leftturn()
{
    digitalWrite(motorr1, HIGH);
    digitalWrite(motorr2, LOW);
    digitalWrite(motorl1, LOW);
    digitalWrite(motorl2, LOW);
    analogWrite(PWMA, 50);
    analogWrite(PWMB, 220);
}

void stop()
{
    digitalWrite(motorl1, LOW);
    digitalWrite(motorl2, LOW);
    digitalWrite(motorr1, LOW);
    digitalWrite(motorr2, LOW);
    analogWrite(PWMA, 0);
    analogWrite(PWMB, 0);
}

void inch()
{
    motors.forward();
    motors.setSpeedA(200);
    motors.setSpeedB(200);
    delay(750);
    motors.forward();
    motors.setSpeedA(0);
    motors.setSpeedB(0);
    delay(500);
}

void new_rinch()
{
    motors.forward();
    motors.setSpeedA(170);
    motors.setSpeedB(228);
    delay(2400);
    motors.forward();
    motors.setSpeedA(0);
    motors.setSpeedB(0);
    delay(1200);
}

void event_e(int seconds){
    forward_for_event(seconds);
    stop_for_event(1000);
    return;
}

void stop_for_event(int seconds)
{
    unsigned long startTime = millis();
    unsigned long duration = seconds;
    while (millis() - startTime < duration)
    {
        motors.stop();
        tone(buzzer, 300, 1000);
        delay(1000);
        noTone(buzzer);
    }
    Serial.println("Out Stop");
    return;
}

void forward_for_event(int seconds)
{
    unsigned long startTime = millis();
    unsigned long duration = seconds;

    while (millis() - startTime < duration)
    {
        readsensors();
        flag = 0;

        if (((centerR == HIGH && leftR == LOW && rightR == LOW) || (centerR == HIGH && leftR == HIGH && rightR == HIGH)) && flag == 0)
        {
            motors.forward();
            motors.setSpeedA(80);
            motors.setSpeedB(80);
            flag = 1;
        }
        else if ((centerR == LOW && leftR == LOW && rightR == HIGH && flag == 0) || (leftR == LOW && centerR == HIGH && rightR == HIGH && flag == 0))
        {
            rightturn();
            flag = 1;
        }
        else if ((centerR == LOW && rightR == LOW && leftR == HIGH && flag == 0) || (leftR == HIGH && centerR == HIGH && rightR == LOW && flag == 0))
        {
            leftturn();
            flag = 1;
        }
        else if (centerR == LOW && leftR == LOW && rightR == LOW && flag == 0)
        {
            if (leftfarR == LOW && rightfarR == LOW)
            {
                forward();
                flag = 1;
            }
            else if (leftfarR == LOW && rightfarR == HIGH)
            {
                leftturn();
                flag = 1;
            }
            else if (leftfarR == HIGH && rightfarR == LOW)
            {
                rightturn();
                flag = 1;
            }
        }
    }
    return;
}

void dynamic_right(){
    do{
        readsensors();
        rightturn();
    }while(rightR != HIGH);

    do{
        readsensors();
        rightturn();
    }while(centerR != HIGH);
}

void dynamic_left(){
    do{
        readsensors();
        leftturn();
    }while(leftR != HIGH);

    do{
        readsensors();
        leftturn();
    }while(centerR != HIGH);
}

void node(char element, char next_element)
{
    tone(buzzer, 100, 1000);
    noTone(buzzer);

    inch();
    
    if(element == 'f')
    {
        Serial.printf("Command: f \n");
        Serial.printf("About to return\n");
    }

    if(element == 'r')
    {
        Serial.printf("Command: r \n");
        rinch();
        delay(1000);
        dynamic_right();
    }

    if(element == 'l')
    {
        Serial.printf("Command: l \n");
        leftturn();
        dynamic_left();
    }

    // Event conditions
    if(next_element == 'e')
    {
        Serial.printf("Command: e");
        event_e(1350);
        counter++;
        return;
    }

    if (next_element == 'a')
    {
        Serial.printf("Command: a");
        event_e(2850);
        counter++;
        return;
    }

    if (next_element == 'c')
    {
        Serial.printf("Command: c");
        event_e(6000);
        counter++;
        return;
    }
    
    if (next_element == 'x')
    {
        Serial.printf("Command: x");
        event_e(3500);
        counter++;
        return;
    }
    if (next_element == 'y')
    {
        Serial.printf("Command: y");
        event_e(7500);
        counter++;
        return;
    }

    if (next_element == 'b'){
    digitalWrite(led, HIGH);
    delay(1000);
    tone(buzzer, 300,1000);
    digitalWrite(led, LOW);
    noTone(buzzer);
    readsensors();
    inch();
    if(counter == 3 || counter == 5 || counter == 6 || counter == 8)
      {
        do{
        readsensors();
        rightturn();
        }while(rightR != HIGH);

        do{
        readsensors();
        rightturn();
        }while(centerR != HIGH);

        backward();

        do{
        readsensors();
        rightturn();
        }while(rightR != HIGH);

        do{
        readsensors();
        rightturn();
        }while(centerR != HIGH);  
    }

    // Final Stop
    if (next_element == 'z')
    {
        Serial.printf("Command: z");
        new_rinch();
        motors.stop();
        tone(buzzer, 1000, 5000);
        delay(5000);
        noTone(buzzer);
        exit(1);
    }

    Serial.println("Out of Node Function");
    return;
}
}

void loop()
{
    // Checks if Bluetooth data is available for the first time
    if(bt_counter == 0) 
    {
        // Waits until Bluetooth data becomes available
        while(!SerialBT.available())
        {
            if(SerialBT.available()){
                break;
            }
        }
        
        // Reads Bluetooth data until 'q' is encountered
        if(SerialBT.available() != 'q') 
        {
            queue = SerialBT.readStringUntil('q'); 
            loop_counter++;
        }
        
        // Copies the characters from the Bluetooth queue to the charArray
        for(int m = 0; m < queue.length(); m++)
        {
            charArray[m] = queue.charAt(m);
        }
        bt_counter++;
    }

    // Reads sensor data
    readsensors();
    flag = 0;

    // Checks the line following conditions and takes appropriate actions
    if (centerR == HIGH && leftR == LOW && rightR == LOW && flag == 0)
    {
        forward();
        flag = 1;
    }
    else if ((centerR == LOW && leftR == LOW && rightR == HIGH && flag == 0) || (leftR == LOW && centerR == HIGH && rightR == HIGH && flag == 0))
    {
        rightturn();
        flag = 1;
    }
    else if ((centerR == LOW && rightR == LOW && leftR == HIGH && flag == 0) || (leftR == HIGH && centerR == HIGH && rightR == LOW && flag == 0))
    {
        leftturn();
        flag = 1;
    }
    else if (centerR == LOW && leftR == LOW && rightR == LOW && flag == 0)
    {
        if ((leftfarR == LOW && rightfarR == LOW) || (leftfarR == HIGH && rightfarR == HIGH))
        {
            forward();
            flag = 1;
        }
        else if (leftfarR == LOW && rightfarR == HIGH)
        {
            leftturn();
            flag = 1;
        }
        else if (leftfarR == HIGH && rightfarR == LOW)
        {
            rightturn();
            flag = 1;
        }
    }
    else if (((centerR == HIGH && leftR == HIGH && rightR == HIGH && leftfarR == HIGH) || (centerR == HIGH && leftR == HIGH && rightR == HIGH && rightfarR == HIGH) || (centerR == HIGH && leftR == HIGH && rightR == HIGH)) && flag == 0 ) 
    {
        // Increment the counter and determine the next action based on the elements in charArray
        counter += 1;
        Serial.println(counter);
        int array_counter_event = counter + 1;
        char array_element_event_checker = charArray[array_counter_event];
        node(charArray[counter], array_element_event_checker);
        flag = 1;
        Serial.println("Out of Node Detection");
        return;
    }
}

