#pragma once
#include "TimerCounter.hpp" 
#include <iostream> 
using namespace std;
TimerCounter::TimerCounter(void){
    QueryPerformanceFrequency(&freq);//获取主机CPU时钟频率
}
TimerCounter::~TimerCounter(void)
{}
void TimerCounter::Start(){
    QueryPerformanceCounter(&startCount);//开始计时
}
void TimerCounter::Stop(){
    QueryPerformanceCounter(&endCount);//停止计时

    dbTime = ((double)endCount.QuadPart - (double)startCount.QuadPart) / (double)freq.QuadPart;//获取时间差

}
