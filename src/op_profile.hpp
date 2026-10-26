#pragma once
#include "znn.hpp"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>

namespace znn { namespace profiling {
// Opt-in host wall time, inclusive of nested dispatch. GPU values measure
// submission, not device completion: use a GPU timeline to attribute kernels.
struct Stats {
    struct Row { uint64_t calls=0; double total=0, maximum=0; };
    std::string path;
    std::map<std::string,Row> rows;
    std::mutex mutex;
    Stats() { if(const char* p=std::getenv("ZNN_PROFILE_FILE")) path=p; }
    ~Stats() {
        if(path.empty())return;
        std::ofstream f(path);
        f<<"device,operator,input_shapes,calls,host_total_ms,host_max_ms\n";
        for(const auto& e:rows)f<<e.first<<','<<e.second.calls<<','<<e.second.total<<','<<e.second.maximum<<'\n';
    }
};
inline Stats& stats(){static Stats s;return s;}
class Scope {
    using Clock=std::chrono::steady_clock;
    Stats* owner=nullptr;
    std::string key;
    Clock::time_point start;
public:
    Scope(const char* op,Span<const Tensor*> inputs) {
        auto& s=stats();if(s.path.empty())return;owner=&s;
        std::ostringstream k;k<<int(inputs[0]->device())<<','<<op<<',';
        for(size_t i=0;i<inputs.size();++i){if(i)k<<';';for(int64_t d=0;d<inputs[i]->dim_count();++d){if(d)k<<'x';k<<inputs[i]->size(d);}}
        key=k.str();start=Clock::now();
    }
    ~Scope() {
        if(!owner)return;
        const double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        std::lock_guard<std::mutex> lock(owner->mutex);
        auto& r=owner->rows[key];++r.calls;r.total+=ms;if(ms>r.maximum)r.maximum=ms;
    }
};
}}
