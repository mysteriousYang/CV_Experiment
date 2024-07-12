#pragma once
#include<Windows.h>
class TimerCounter
{
public:
    TimerCounter(void);//构造函数
    ~TimerCounter(void);//析构函数
private:
    LARGE_INTEGER startCount;//记录开始时间

    LARGE_INTEGER endCount;//记录结束时间

    LARGE_INTEGER freq;//本机CPU时钟频率
public:
    double dbTime;//程序运行的时间保存在这里

public:
    void Start();//被测程序开始点处开始计时
    void Stop();//被测程序结束点处结束计时
};
