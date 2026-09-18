// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <any>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

// Helper functions for extracting values from std::any with type conversion

// Convert hex string to bytes or use bytes directly
inline std::vector<uint8_t> extract_or_convert_bytes(const std::any& value) {
    // Try as vector<uint8_t> first
    if (value.type() == typeid(std::vector<uint8_t>)) {
        return std::any_cast<std::vector<uint8_t>>(value);
    }
    // Try as string (hex)
    if (value.type() == typeid(std::string)) {
        extern std::vector<uint8_t> hex_to_bytes(const std::string& hex);
        return hex_to_bytes(std::any_cast<std::string>(value));
    }
    throw std::bad_any_cast();
}

// Extract numeric value with type flexibility
template<typename T>
T extract_numeric_flexible(const std::any& value, T default_val = 0) {
    // Direct type match
    if (value.type() == typeid(T)) {
        return std::any_cast<T>(value);
    }
    // Try various numeric types
    try {
        if (value.type() == typeid(int)) return static_cast<T>(std::any_cast<int>(value));
        if (value.type() == typeid(int64_t)) return static_cast<T>(std::any_cast<int64_t>(value));
        if (value.type() == typeid(uint32_t)) return static_cast<T>(std::any_cast<uint32_t>(value));
        if (value.type() == typeid(uint64_t)) return static_cast<T>(std::any_cast<uint64_t>(value));
        if (value.type() == typeid(float)) return static_cast<T>(std::any_cast<float>(value));
        if (value.type() == typeid(double)) return static_cast<T>(std::any_cast<double>(value));
        // Try string conversion
        if (value.type() == typeid(std::string)) {
            const std::string& str = std::any_cast<std::string>(value);
            if constexpr (std::is_integral_v<T>) {
                return static_cast<T>(std::stoll(str));
            } else {
                return static_cast<T>(std::stod(str));
            }
        }
    } catch (...) {}
    return default_val;
}

// Extract vector with type flexibility
template<typename T>
std::vector<T> extract_vector_flexible(const std::any& value) {
    // Direct match
    if (value.type() == typeid(std::vector<T>)) {
        return std::any_cast<std::vector<T>>(value);
    }
    
    // Try conversions for numeric vectors
    if constexpr (std::is_arithmetic_v<T>) {
        // uint32_t <-> int32_t conversion
        if (value.type() == typeid(std::vector<int32_t>) && typeid(T) == typeid(uint32_t)) {
            const auto& vec = std::any_cast<std::vector<int32_t>>(value);
            return std::vector<T>(vec.begin(), vec.end());
        }
        if (value.type() == typeid(std::vector<uint32_t>) && typeid(T) == typeid(int32_t)) {
            const auto& vec = std::any_cast<std::vector<uint32_t>>(value);
            return std::vector<T>(vec.begin(), vec.end());
        }
    }
    
    throw std::bad_any_cast();
}

// Extract 2D matrix with type flexibility
template<typename T>
std::vector<std::vector<T>> extract_matrix_flexible(const std::any& value) {
    // Direct match
    if (value.type() == typeid(std::vector<std::vector<T>>)) {
        return std::any_cast<std::vector<std::vector<T>>>(value);
    }
    
    // Try conversions for numeric matrices
    if constexpr (std::is_arithmetic_v<T>) {
        // uint32_t <-> int32_t conversion
        if (value.type() == typeid(std::vector<std::vector<int32_t>>) && typeid(T) == typeid(uint32_t)) {
            const auto& mat = std::any_cast<std::vector<std::vector<int32_t>>>(value);
            std::vector<std::vector<T>> result;
            result.reserve(mat.size());
            for (const auto& row : mat) {
                result.emplace_back(row.begin(), row.end());
            }
            return result;
        }
        if (value.type() == typeid(std::vector<std::vector<uint32_t>>) && typeid(T) == typeid(int32_t)) {
            const auto& mat = std::any_cast<std::vector<std::vector<uint32_t>>>(value);
            std::vector<std::vector<T>> result;
            result.reserve(mat.size());
            for (const auto& row : mat) {
                result.emplace_back(row.begin(), row.end());
            }
            return result;
        }
    }
    
    throw std::bad_any_cast();
}