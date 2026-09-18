#include "stratum.h"
#include "api.h"
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::string error;
    meow::StatsHttpServer api;
    if (!api.start(std::string("127.0.0.1:") + argv[2], [] { return "{\"test\":true}"; }, error)) return 3;
    meow::PoolUrl url;
    if (!meow::PoolUrl::parse(std::string("stratum+tcp://127.0.0.1:") + argv[1], url, error)) return 4;
    meow::StratumClient client;
    std::atomic<bool> accepted{false};
    meow::StratumCallbacks callbacks;
    callbacks.on_job = [&](const meow::PoolJob& job) { client.submit(job.job_id, 42, "TEST", "ff", 1000); };
    callbacks.on_submit_result = [&](bool ok, int, const std::string&, int) { accepted = ok; };
    client.configure({url}, "test.worker", "x", "supr-meow-tsc/0.7.0", callbacks);
    if (!client.start()) return 5;
    for (int i=0; i<100 && !accepted; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    client.stop();
    api.stop();
    if (!accepted) return 6;
    std::cout << "Stratum subscribe/authorize/job/submit and shutdown passed\n";
}
