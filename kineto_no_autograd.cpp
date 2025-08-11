// custom_profiler.cpp
// cmake .. -DCMAKE_PREFIX_PATH=/home/admin1/.local/lib/python3.10/site-packages/torch/share/cmake
// make -j$(nproc)

#include <torch/extension.h>
#include <ATen/record_function.h>
#include <fstream>
#include <chrono>
#include <mutex>
#include <thread>
#include <sstream>
#include <memory>
#include <unordered_map>
#include <sys/types.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <cstdlib> // getenv
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <time.h>
#include <vector>

struct Event {
    std::string name;
    std::string scope;
    int pid;
    unsigned int tid;
    long long ts;           // microseconds since profiler start (wall)
    long long dur;          // microseconds duration (wall)
    long long cpu_time_us;  // inclusive CPU time (microseconds)
    long long self_cpu_us;  // exclusive/self CPU time (microseconds)
    int external_id;
    std::vector<std::vector<int64_t>> input_dims;
    std::vector<std::string> input_types;
    long long mem_diff_kb; // memory delta in KB
};

static std::vector<Event> events;
static std::mutex events_mutex;
static bool is_profiling = false;
static int external_id_counter = 0;
static std::chrono::steady_clock::time_point start_time;

struct ProfilerContext : public at::ObserverContext {
    std::chrono::steady_clock::time_point start;
    int external_id;
    std::string op_name;
    at::RecordScope scope;
    std::vector<std::vector<int64_t>> input_dims;
    std::vector<std::string> input_types;
    long long mem_before_kb;

    // CPU timestamps (per-thread)
    struct timespec cpu_before_ts;

    // accumulate children's CPU/us to compute self time
    long long child_cpu_us = 0;
    long long child_wall_us = 0;
};

// thread-local stack for nesting -> moved to file scope to avoid lambda capture warnings
static thread_local std::vector<ProfilerContext*> ctx_stack;

static inline std::string recordScopeToString(at::RecordScope scope) {
    switch (scope) {
        case at::RecordScope::FUNCTION: return "FUNCTION";
        case at::RecordScope::BACKWARD_FUNCTION: return "BACKWARD_FUNCTION";
        case at::RecordScope::TORCHSCRIPT_FUNCTION: return "TORCHSCRIPT_FUNCTION";
        case at::RecordScope::USER_SCOPE: return "USER_SCOPE";
        default: return "UNKNOWN_SCOPE";
    }
}

static inline bool debug_enabled() {
    const char* env = std::getenv("CUSTOM_PROFILER_DEBUG");
    return env && std::string(env) == "1";
}

// Read current RSS (resident set size) in KB from /proc/self/statm
static long long get_rss_kb() {
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long size_pages = 0, rss_pages = 0;
    if (fscanf(f, "%ld %ld", &size_pages, &rss_pages) < 2) {
        std::fclose(f);
        return 0;
    }
    std::fclose(f);
    long page_size_kb = sysconf(_SC_PAGESIZE) / 1024;
    return static_cast<long long>(rss_pages) * page_size_kb;
}

// get current thread CPU time in microseconds
static inline long long get_thread_cpu_us_from_timespec(const struct timespec &ts) {
    return static_cast<long long>(ts.tv_sec) * 1000000LL + static_cast<long long>(ts.tv_nsec) / 1000LL;
}
static inline struct timespec now_thread_cpu_timespec() {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts;
}

void start_profiler() {
    {
        std::lock_guard<std::mutex> lock(events_mutex);
        events.clear();
        external_id_counter = 0;
        is_profiling = true;
        start_time = std::chrono::steady_clock::now();
    }

    static bool callback_registered = false;
    if (callback_registered) return;

    at::addGlobalCallback(at::RecordFunctionCallback(
        // start callback
        [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
            if (!is_profiling) return nullptr;

            auto ctx = std::make_unique<ProfilerContext>();
            ctx->start = std::chrono::steady_clock::now();
            ctx->external_id = ++external_id_counter;
            ctx->op_name = fn.name() ? std::string(fn.name()) : std::string("unknown_op");
            ctx->scope = fn.scope();

            // capture mem and cpu before
            ctx->mem_before_kb = get_rss_kb();
            ctx->cpu_before_ts = now_thread_cpu_timespec();

            // Safely capture inputs only in start callback
            try {
                if (!fn.inputs().empty()) {
                    for (auto& iv : fn.inputs()) {
                        if (iv.isTensor()) {
                            try {
                                auto t = iv.toTensor();
                                if (t.defined()) {
                                    ctx->input_types.push_back(c10::toString(t.scalar_type()));
                                    ctx->input_dims.push_back(std::vector<int64_t>(t.sizes().begin(), t.sizes().end()));
                                }
                            } catch (const c10::Error &e) {
                                if (debug_enabled()) {
                                    std::cerr << "[profiler debug] input->toTensor() failed for op " << ctx->op_name
                                              << " : " << e.what() << "\n";
                                }
                            }
                        }
                    }
                }
            } catch (const c10::Error &e) {
                if (debug_enabled()) {
                    std::cerr << "[profiler debug] fn.inputs() threw: " << e.what()
                              << " -- skipping inputs for op: " << ctx->op_name << "\n";
                }
            }

            // push onto thread-local stack for nesting tracking
            ctx_stack.push_back(ctx.get());

            return ctx;
        },
        // end callback
        [](const at::RecordFunction& fn, at::ObserverContext* context) {
            if (!is_profiling || !context) return;
            auto* ctx = static_cast<ProfilerContext*>(context);

            auto end = std::chrono::steady_clock::now();
            auto ts = std::chrono::duration_cast<std::chrono::microseconds>(ctx->start - start_time).count();
            auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - ctx->start).count();

            // memory and cpu after
            long long mem_after_kb = get_rss_kb();
            struct timespec cpu_after_ts = now_thread_cpu_timespec();

            long long mem_diff_kb = mem_after_kb - ctx->mem_before_kb;
            long long cpu_before_us = get_thread_cpu_us_from_timespec(ctx->cpu_before_ts);
            long long cpu_after_us = get_thread_cpu_us_from_timespec(cpu_after_ts);
            long long cpu_time_us = std::max(0LL, cpu_after_us - cpu_before_us);

            // compute self/inclusive times using the stack:
            long long self_cpu_us = cpu_time_us - ctx->child_cpu_us;
            if (self_cpu_us < 0) self_cpu_us = 0;

            Event e;
            e.name = ctx->op_name;
            e.scope = recordScopeToString(ctx->scope);
            e.pid = ::getpid();
            e.tid = static_cast<unsigned int>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
            e.ts = ts;
            e.dur = dur;
            e.cpu_time_us = cpu_time_us;   // inclusive
            e.self_cpu_us = self_cpu_us;   // exclusive/self
            e.external_id = ctx->external_id;
            e.input_dims = std::move(ctx->input_dims);
            e.input_types = std::move(ctx->input_types);
            e.mem_diff_kb = mem_diff_kb;

            {
                std::lock_guard<std::mutex> lock(events_mutex);
                if (!e.name.empty()) {
                    events.push_back(std::move(e));
                }
            }

            // pop this ctx from stack (top should be this)
            if (!ctx_stack.empty()) {
                if (ctx_stack.back() == ctx) {
                    ctx_stack.pop_back();
                } else {
                    // defensive removal
                    for (auto it = ctx_stack.rbegin(); it != ctx_stack.rend(); ++it) {
                        if (*it == ctx) {
                            ctx_stack.erase(std::next(it).base());
                            break;
                        }
                    }
                }
            }

            // add this inclusive cpu/wall to parent child accumulators
            if (!ctx_stack.empty()) {
                ProfilerContext* parent = ctx_stack.back();
                parent->child_cpu_us += cpu_time_us;
                parent->child_wall_us += dur;
            }

            if (debug_enabled()) {
                std::cout << "[Profiler] Captured event: " << ctx->op_name
                          << ", dur(us): " << dur << ", ts(us): " << ts
                          << ", mem_diff_kb: " << mem_diff_kb
                          << ", cpu_time_us: " << cpu_time_us
                          << ", self_cpu_us: " << self_cpu_us
                          << ", thread: " << std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()))
                          << "\n";
            }
        }
    ).needsInputs(true)
     .scopes({
        at::RecordScope::FUNCTION,
        at::RecordScope::BACKWARD_FUNCTION,
        at::RecordScope::TORCHSCRIPT_FUNCTION,
        at::RecordScope::USER_SCOPE
     }));

    callback_registered = true;
}

void stop_profiler(const std::string& filename) {
    is_profiling = false;

    // Snapshot events under lock
    std::vector<Event> snapshot;
    {
        std::lock_guard<std::mutex> lock(events_mutex);
        snapshot = events; // copy
    }

    // Compute totals for CPU times
    long long total_self_cpu_us = 0;
    for (const auto &e : snapshot) {
        total_self_cpu_us += e.self_cpu_us;
    }
    if (total_self_cpu_us == 0) total_self_cpu_us = 1; // avoid div by zero

    // Write trace JSON compatible with chrome tracing, include mem and cpu% (compute cpu% from total_self_cpu_us)
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Failed to open " << filename << " for writing\n";
        return;
    }

    out << "{\n";
    out << "  \"displayTimeUnit\": \"ms\",\n";
    out << "  \"traceEvents\": [\n";

    int pid = ::getpid();
    out << "    {\"name\": \"process_name\", \"ph\": \"M\", \"pid\": " << pid
        << ", \"args\": {\"name\": \"Custom Profiler\"}},\n";

    std::unordered_map<unsigned int, bool> thread_ids;
    for (const auto& e : snapshot) {
        if (thread_ids.find(e.tid) == thread_ids.end()) {
            out << "    {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": " << e.pid
                << ", \"tid\": " << e.tid << ", \"args\": {\"name\": \"WorkerThread\"}},\n";
            thread_ids[e.tid] = true;
        }
    }

    for (size_t i = 0; i < snapshot.size(); ++i) {
        const auto& e = snapshot[i];
        double cpu_total_pct = (static_cast<double>(e.cpu_time_us) / static_cast<double>(total_self_cpu_us)) * 100.0;
        double cpu_self_pct  = (static_cast<double>(e.self_cpu_us) / static_cast<double>(total_self_cpu_us)) * 100.0;

        out << "    {\"ph\": \"X\", \"cat\": \"cpu_op\", \"name\": \"" << e.name << "\", ";
        out << "\"pid\": " << e.pid << ", \"tid\": " << e.tid << ", ";
        out << "\"ts\": " << e.ts << ", \"dur\": " << e.dur << ", ";
        out << "\"args\": {\"External id\": " << e.external_id << ", \"Scope\": \"" << e.scope << "\""
            << ", \"Mem_KB\": " << e.mem_diff_kb
            << ", \"CPU_total_pct\": " << std::fixed << std::setprecision(3) << cpu_total_pct
            << ", \"CPU_self_pct\": " << std::fixed << std::setprecision(3) << cpu_self_pct
            << "}}\n";
        if (i + 1 < snapshot.size()) out << ",";
    }

    out << "  ]\n";
    out << "}\n";
    out.close();

    // Summary table: aggregate per-op totals (calls, total dur, total mem, total_cpu_us, total_self_cpu_us)
    struct Agg { int calls=0; long long total_dur=0; long long total_mem=0; long long total_cpu_us=0; long long total_self_cpu_us=0; };
    std::unordered_map<std::string, Agg> agg;
    for (const auto& e : snapshot) {
        auto &a = agg[e.name];
        a.calls += 1;
        a.total_dur += e.dur;
        a.total_mem += e.mem_diff_kb;
        a.total_cpu_us += e.cpu_time_us;
        a.total_self_cpu_us += e.self_cpu_us;
    }

    // sort by total duration
    std::vector<std::pair<std::string, Agg>> sorted_ops(agg.begin(), agg.end());
    std::sort(sorted_ops.begin(), sorted_ops.end(), [](const auto& A, const auto& B){
        return A.second.total_dur > B.second.total_dur;
    });

    // print header (with memory and cpu%)
    size_t max_name_len = 23;
    for (const auto &p : sorted_ops) if (p.first.size() + 2 > max_name_len) max_name_len = p.first.size() + 2;
    int calls_col = 12, time_col = 14, avg_col = 14, mem_col = 12, cpu_self_col = 12, cpu_total_col = 12;

    std::string sep = std::string(max_name_len, '-') + "  " +
                      std::string(calls_col, '-') + "  " +
                      std::string(time_col, '-') + "  " +
                      std::string(avg_col, '-') + "  " +
                      std::string(mem_col, '-') + "  " +
                      std::string(cpu_self_col, '-') + "  " +
                      std::string(cpu_total_col, '-');

    std::cout << sep << "\n";
    std::cout << std::left << std::setw(max_name_len) << "Name" << "  "
              << std::right << std::setw(calls_col) << "Calls" << "  "
              << std::setw(time_col) << "Time (ms)" << "  "
              << std::setw(avg_col) << "Avg (ms)" << "  "
              << std::setw(mem_col) << "Mem (KB)" << "  "
              << std::setw(cpu_self_col) << "Self CPU %" << "  "
              << std::setw(cpu_total_col) << "CPU total %" << "\n";
    std::cout << sep << "\n";

    // compute global total self CPU for percentage (again)
    long long global_total_self = 0;
    for (const auto &p : sorted_ops) global_total_self += p.second.total_self_cpu_us;
    if (global_total_self == 0) global_total_self = 1;

    for (const auto &p : sorted_ops) {
        const auto &op = p.first;
        const auto &a = p.second;
        double total_ms = static_cast<double>(a.total_dur) / 1000.0;
        double avg_ms = (a.calls > 0) ? (total_ms / a.calls) : 0.0;
        long long avg_mem_kb = (a.calls > 0) ? (a.total_mem / a.calls) : 0;
        double self_pct = (static_cast<double>(a.total_self_cpu_us) / static_cast<double>(global_total_self)) * 100.0;
        double total_pct = (static_cast<double>(a.total_cpu_us) / static_cast<double>(global_total_self)) * 100.0;

        std::cout << std::left << std::setw(max_name_len) << op << "  "
                  << std::right << std::setw(calls_col) << a.calls << "  "
                  << std::setw(time_col) << std::fixed << std::setprecision(3) << total_ms << "  "
                  << std::setw(avg_col) << std::fixed << std::setprecision(6) << avg_ms << "  "
                  << std::setw(mem_col) << avg_mem_kb << "  "
                  << std::setw(cpu_self_col) << std::fixed << std::setprecision(2) << self_pct << "  "
                  << std::setw(cpu_total_col) << std::fixed << std::setprecision(2) << total_pct
                  << "\n";
    }

    std::cout << sep << "\n";
    std::cout << "Trace saved to " << filename << "\n";
}

PYBIND11_MODULE(custom_profiler, m) {
    m.def("start_profiler", &start_profiler, "Start custom profiler");
    m.def("stop_profiler", &stop_profiler, "Stop custom profiler and save trace");
}
