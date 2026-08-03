#include "device.h"

#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>

namespace meow {

bool parse_device_spec(const std::string& spec, std::vector<int>& out, std::string& error) {
    out.clear();
    std::string s;
    for (char c : spec) if (!isspace(static_cast<unsigned char>(c))) s += c;
    if (s.empty()) return true;                    // no -d  → all devices

    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) { error = "empty entry in device list (check for a stray comma)"; return false; }
        for (char c : tok) {
            if (!isdigit(static_cast<unsigned char>(c))) {
                error = "device list must be numbers separated by commas, got '" + tok + "'";
                return false;
            }
        }
        const int id = std::stoi(tok);
        if (std::find(out.begin(), out.end(), id) != out.end()) {
            error = "device " + tok + " listed twice";
            return false;
        }
        out.push_back(id);
    }
    return true;
}

std::string DeviceManager::sm_arch_name(int major, int minor) {
    const int sm = major * 10 + minor;
    switch (sm) {
        case 120: return "Blackwell (sm_120)";
        case 90:  return "Hopper (sm_90)";
        case 89:  return "Ada (sm_89)";
        case 86:  return "Ampere (sm_86)";
        case 80:  return "Ampere (sm_80)";
        case 75:  return "Turing (sm_75)";
        default: {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "sm_%d%d", major, minor);
            return buf;
        }
    }
}

bool DeviceManager::init(const std::string& spec, std::string& error) {
    int count = 0;
    // Report what CUDA actually said. "No device" and "the driver rejected the
    // call" are different problems with different fixes, and a rig where
    // nvidia-smi works but CUDA does not is exactly the case the generic
    // message cannot distinguish — nvidia-smi uses NVML, not CUDA.
    const cudaError_t rc = cudaGetDeviceCount(&count);
    if (rc != cudaSuccess || count == 0) {
        error = "no usable CUDA device: " + std::string(cudaGetErrorName(rc)) + " — " +
                cudaGetErrorString(rc) + " (visible devices: " + std::to_string(count) + ")";
        return false;
    }

    std::vector<int> wanted;
    if (!parse_device_spec(spec, wanted, error)) return false;
    if (wanted.empty()) {
        for (int i = 0; i < count; ++i) wanted.push_back(i);
    }
    for (int id : wanted) {
        if (id < 0 || id >= count) {
            error = "device " + std::to_string(id) + " does not exist (" +
                    std::to_string(count) + " device(s) present: 0.." + std::to_string(count - 1) + ")";
            return false;
        }
    }

    nvml_ready_ = (nvmlInit_v2() == NVML_SUCCESS);

    for (int id : wanted) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, id) != cudaSuccess) {
            error = "cannot query CUDA device " + std::to_string(id);
            return false;
        }
        DeviceInfo d;
        d.index      = id;
        d.name       = p.name;
        d.sm_major   = p.major;
        d.sm_minor   = p.minor;
        d.vram_total = p.totalGlobalMem;
        d.pci_bus    = p.pciBusID;   // decimal — HiveOS bus_numbers wants this

        size_t freeb = 0, totalb = 0;
        if (cudaSetDevice(id) == cudaSuccess && cudaMemGetInfo(&freeb, &totalb) == cudaSuccess) {
            d.vram_free = freeb;
        }
        if (nvml_ready_) {
            nvmlDevice_t h{};
            if (nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned>(id), &h) == NVML_SUCCESS) {
                d.nvml_ok = true;
                char uuid[NVML_DEVICE_UUID_BUFFER_SIZE] = {0};
                if (nvmlDeviceGetUUID(h, uuid, sizeof(uuid)) == NVML_SUCCESS) d.uuid = uuid;
            }
        }
        devices_.push_back(d);
    }
    return true;
}

DeviceTelemetry DeviceManager::telemetry(int index) const {
    DeviceTelemetry t;
    t.index = index;
    if (!nvml_ready_) return t;

    nvmlDevice_t h{};
    if (nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned>(index), &h) != NVML_SUCCESS) return t;

    unsigned u = 0;
    if (nvmlDeviceGetTemperature(h, NVML_TEMPERATURE_GPU, &u) == NVML_SUCCESS) t.temp_c = static_cast<int>(u);
    if (nvmlDeviceGetFanSpeed(h, &u) == NVML_SUCCESS)                          t.fan_pct = static_cast<int>(u);
    if (nvmlDeviceGetPowerUsage(h, &u) == NVML_SUCCESS)                        t.power_w = static_cast<int>(u / 1000);
    if (nvmlDeviceGetEnforcedPowerLimit(h, &u) == NVML_SUCCESS)                t.power_limit_w = static_cast<int>(u / 1000);
    if (nvmlDeviceGetClockInfo(h, NVML_CLOCK_GRAPHICS, &u) == NVML_SUCCESS)    t.clock_core = static_cast<int>(u);
    if (nvmlDeviceGetClockInfo(h, NVML_CLOCK_MEM, &u) == NVML_SUCCESS)         t.clock_mem = static_cast<int>(u);

    nvmlUtilization_t util{};
    if (nvmlDeviceGetUtilizationRates(h, &util) == NVML_SUCCESS) {
        t.util_gpu = static_cast<int>(util.gpu);
        t.util_mem = static_cast<int>(util.memory);
    }
    nvmlMemory_t mem{};
    if (nvmlDeviceGetMemoryInfo(h, &mem) == NVML_SUCCESS) t.vram_used = mem.used;

    // Memory-junction temperature is newer and not on every card; absence is
    // normal, not an error.
#ifdef NVML_TEMPERATURE_MEM
    if (nvmlDeviceGetTemperature(h, NVML_TEMPERATURE_MEM, &u) == NVML_SUCCESS) t.temp_mem_c = static_cast<int>(u);
#endif

    t.valid = true;
    return t;
}

bool DeviceManager::apply_tuning(int index, const DeviceTuning& tun, std::string& error) {
    if (!nvml_ready_) { error = "NVML unavailable — cannot tune"; return false; }
    nvmlDevice_t h{};
    if (nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned>(index), &h) != NVML_SUCCESS) {
        error = "no NVML handle for device " + std::to_string(index);
        return false;
    }
    bool touched = false;

    if (tun.power_limit_w != DeviceTuning::UNSET) {
        const nvmlReturn_t r = nvmlDeviceSetPowerManagementLimit(h, static_cast<unsigned>(tun.power_limit_w) * 1000u);
        if (r != NVML_SUCCESS) { error = std::string("power limit: ") + nvmlErrorString(r); return false; }
        touched = true;
    }
    if (tun.core_offset_mhz != DeviceTuning::UNSET) {
        const nvmlReturn_t r = nvmlDeviceSetGpcClkVfOffset(h, tun.core_offset_mhz);
        if (r != NVML_SUCCESS) { error = std::string("core offset: ") + nvmlErrorString(r); return false; }
        touched = true;
    }
    if (tun.mem_offset_mhz != DeviceTuning::UNSET) {
        const nvmlReturn_t r = nvmlDeviceSetMemClkVfOffset(h, tun.mem_offset_mhz);
        if (r != NVML_SUCCESS) { error = std::string("memory offset: ") + nvmlErrorString(r); return false; }
        touched = true;
    }
    if (tun.lock_core_mhz != DeviceTuning::UNSET) {
        const nvmlReturn_t r = nvmlDeviceSetGpuLockedClocks(h, 0u, static_cast<unsigned>(tun.lock_core_mhz));
        if (r != NVML_SUCCESS) { error = std::string("locked clocks: ") + nvmlErrorString(r); return false; }
        touched = true;
    }
    if (tun.fan_pct != DeviceTuning::UNSET) {
        unsigned fans = 0;
        if (nvmlDeviceGetNumFans(h, &fans) != NVML_SUCCESS) fans = 1;
        for (unsigned f = 0; f < fans; ++f) {
            const nvmlReturn_t r = nvmlDeviceSetFanSpeed_v2(h, f, static_cast<unsigned>(tun.fan_pct));
            if (r != NVML_SUCCESS) { error = std::string("fan: ") + nvmlErrorString(r); return false; }
        }
        touched = true;
    }
    if (touched && std::find(tuned_.begin(), tuned_.end(), index) == tuned_.end()) {
        tuned_.push_back(index);
    }
    return true;
}

void DeviceManager::restore_all() {
    if (!nvml_ready_) return;
    for (int idx : tuned_) {
        nvmlDevice_t h{};
        if (nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned>(idx), &h) != NVML_SUCCESS) continue;
        // Hand the card back: fans to the driver's curve, clocks unlocked.
        // Offsets and power limit are left as set — an operator who asked for
        // them usually wants them to persist for the next run.
        nvmlDeviceSetDefaultFanSpeed_v2(h, 0);
        nvmlDeviceResetGpuLockedClocks(h);
    }
    tuned_.clear();
}

DeviceManager::~DeviceManager() {
    restore_all();
    if (nvml_ready_) nvmlShutdown();
}

}  // namespace meow
