#ifndef __ZCOM_TICK_H__
#define __ZCOM_TICK_H__

#include <cstdint>
//#include <iostream>
#if _WIN32
#include <windows.h>
#endif

namespace zmodule
{
namespace zcom
{

    class tick_time
    {
    public:
        static int64_t now_mills()
        {
            int64_t now = tick_time::now();

#if _WIN32
            LARGE_INTEGER qpfreq;
            QueryPerformanceFrequency(&qpfreq);
            return (now * 1000) / qpfreq.QuadPart;
#elif defined CM_LINUX
            return now / 1000000LL;
#else
            return now / 1000LL;
#endif
        }

        static int64_t now_micros()
        {
            int64_t now = tick_time::now();

#if _WIN32
            LARGE_INTEGER qpfreq;
            QueryPerformanceFrequency(&qpfreq);
            return (now * 1000000) / qpfreq.QuadPart;
#elif defined CM_LINUX
            return now / 1000LL;
#else
            return now;
#endif

        }

        static int64_t now_nanos()
        {
            return now_micros() * 1000;
        }

    private:
        static int64_t now()
        {
#if _WIN32
            // QueryPerformanceCounter returns the value from the TSC which is
            // incremented at the CPU frequency. The algorithm used requires
            // the CPU frequency to be constant. Technology like speed stepping
            // which has variable CPU frequency will therefore yield unpredictable,
            // incorrect time estimations.
            LARGE_INTEGER qpcnt;
            QueryPerformanceCounter(&qpcnt);
            int64_t ticks = qpcnt.QuadPart;
#elif defined CM_LINUX
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            int64_t ticks = 1000000000LL * static_cast<int64_t>(ts.tv_sec) + static_cast<int64_t>(ts.tv_nsec);
#else
            struct timeval tv;
            gettimeofday(&tv, NULL);
            int64_t ticks = 1000000LL * static_cast<int64_t>(tv.tv_sec) + static_cast<int64_t>(tv.tv_usec);
#endif
            return ticks;
        }

    };

}
}

#endif  // end of __ZCOM_TICK_H__
