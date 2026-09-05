#pragma once
#include <string>

namespace meow {
inline constexpr const char* q8_model_identifier =
    "Qwen/Qwen3-8B@9c925d64d72725edaf899c6cb9c377fd0709d9c5";
inline constexpr const char* q8_model_sha256 =
    "30f3a9df384a08453ff0bf5715489e174e986cd033af2a9430e4ddb039cd73c7";
bool verify_q8_model(const std::string& path, std::string& error);
}
