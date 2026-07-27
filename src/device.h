// =============================================================================
// device.h — GPU enumeration, telemetry and clock control for supr-meow-tsc.
//
// NVML gives us everything a miner operator expects to see per card: name,
// VRAM, temperature, fan, core/memory clocks, power draw and utilisation —
// plus the setters for offsets, power limit and fan curve.
//
// Deliberately NVML-only (no nvidia-smi shelling): a miner must not fork a
// process per telemetry tick, and parsing CLI output breaks whenever the driver
// changes its formatting.
//
// Everything degrades gracefully. A rig running an older driver, a container
// without NVML, or a card that refuses clock control must still MINE — telemetry
// is a convenience, mining is the job. Failures are reported once and the miner
// carries on.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace meow {

struct DeviceInfo {
    int         index      = -1;      // CUDA ordinal, what -d refers to
    std::string name;                 // "NVIDIA GeForce RTX 5090"
    std::string uuid;
    int         sm_major   = 0;
    int         sm_minor   = 0;
    size_t      vram_total = 0;       // bytes
    size_t      vram_free  = 0;
    bool        nvml_ok    = false;   // telemetry available for this card
};

struct DeviceTelemetry {
    int      index        = -1;
    int      temp_c       = -1;       // core temperature
    int      temp_mem_c   = -1;       // memory junction, if the card reports it
    int      fan_pct      = -1;       // -1 when the card has no controllable fan
    int      power_w      = -1;
    int      power_limit_w= -1;
    int      clock_core   = -1;       // MHz, current
    int      clock_mem    = -1;       // MHz, current
    int      util_gpu     = -1;       // %
    int      util_mem     = -1;       // %
    size_t   vram_used    = 0;
    bool     valid        = false;
};

// Tuning request. Anything left at the sentinel is not touched — a miner must
// never silently reset a card the operator tuned elsewhere (HiveOS, afterburner).
struct DeviceTuning {
    static constexpr int UNSET = INT32_MIN;
    int core_offset_mhz = UNSET;      // +/- offset applied to the graphics clock
    int mem_offset_mhz  = UNSET;      // +/- offset applied to the memory clock
    int power_limit_w   = UNSET;      // absolute watts
    int fan_pct         = UNSET;      // absolute %, or UNSET to leave on auto
    int lock_core_mhz   = UNSET;      // hard clock lock (locked clocks, not offset)
};

class DeviceManager {
public:
    ~DeviceManager();

    // Enumerate CUDA devices; initialises NVML if present.
    // `spec` is the -d argument: empty = every device, else "0" / "0,1" / "0,2,3".
    // Returns false and fills `error` when the spec names a device that does not
    // exist — a typo must not silently mine on the wrong card.
    bool init(const std::string& spec, std::string& error);

    const std::vector<DeviceInfo>& devices() const { return devices_; }

    // Live telemetry for a selected device. `valid=false` when NVML is absent.
    DeviceTelemetry telemetry(int index) const;

    // Apply tuning. Returns false with `error` describing the FIRST failure; any
    // setting that succeeded stays applied. Most of these need either root or
    // persistence mode — the caller reports the reason rather than dying.
    bool apply_tuning(int index, const DeviceTuning& t, std::string& error);

    // Restore anything we changed (fan back to auto, locked clocks released).
    // Called on shutdown so a rig is not left pinned by a crashed miner.
    void restore_all();

    static std::string sm_arch_name(int major, int minor);

private:
    std::vector<DeviceInfo> devices_;      // only the SELECTED devices
    std::vector<int>        tuned_;        // indices we must restore
    bool                    nvml_ready_ = false;
};

// Parse "-d" into ordinals. Accepts "0", "0,1", " 0 , 2 ". Empty string means
// "all devices" and yields an empty vector (the caller treats that as all).
bool parse_device_spec(const std::string& spec, std::vector<int>& out, std::string& error);

}  // namespace meow
