// SPDX-License-Identifier: Apache-2.0
// any_map_dumper.h
#pragma once

#include <any>
#include <iostream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <cxxabi.h>

// Demangle RTTI name
static std::string demangle(const char* name) {
    int status = 0;
    char* real = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    std::string ret = (status == 0 && real) ? real : name;
    std::free(real);
    return ret;
}

// Helper to dump a flat vector
template<typename T>
static std::string vec_to_string(const std::vector<T>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(v[i]);
    }
    s += "]";
    return s;
}

// Helper to dump a matrix
template<typename T>
static std::string mat_to_string(const std::vector<std::vector<T>>& M) {
    std::string s = "[";
    for (size_t i = 0; i < M.size(); ++i) {
        if (i) s += ", ";
        s += vec_to_string(M[i]);
    }
    s += "]";
    return s;
}

static std::string any_to_string(const std::any& a) {
    const auto &t = a.type();
    std::string tn = demangle(t.name());

    if (t == typeid(int))           return std::to_string(std::any_cast<int>(a));
    if (t == typeid(bool))          return std::any_cast<bool>(a) ? "true" : "false";
    if (t == typeid(std::string))   return "\"" + std::any_cast<std::string>(a) + "\"";

    // integer vectors (prompt_tokens, chosen_tokens, topk_indices)
    if (t == typeid(std::vector<int>)) {
        return vec_to_string(std::any_cast<const std::vector<int>&>(a));
    }
    // float vectors (chosen_probs, sampling_u, softmax_normalizers)
    if (t == typeid(std::vector<float>)) {
        return vec_to_string(std::any_cast<const std::vector<float>&>(a));
    }
    // bool vectors (attention_mask)
    if (t == typeid(std::vector<bool>)) {
        // vector<bool> is specialized; cast to vector<uint8_t> for printing
        const auto &vb = std::any_cast<const std::vector<bool>&>(a);
        std::string s = "[";
        for (size_t i = 0; i < vb.size(); ++i) {
            if (i) s += ", ";
            s += (vb[i] ? "true" : "false");
        }
        s += "]";
        return s;
    }
    // matrix of floats (topk_logits, logsumexp_stats)
    if (t == typeid(std::vector<std::vector<float>>)) {
        return mat_to_string(std::any_cast<const std::vector<std::vector<float>>&>(a));
    }
    // matrix of ints (topk_indices)
    if (t == typeid(std::vector<std::vector<int>>)) {
        return mat_to_string(std::any_cast<const std::vector<std::vector<int>>&>(a));
    }

    // fallback
    return "<" + tn + ">";
}

inline void dump_any_map(const std::unordered_map<std::string,std::any>& m) {
    std::cout << "{\n";
    for (auto const &kv : m) {
        std::cout << "  \"" << kv.first << "\": "
                  << any_to_string(kv.second)
                  << "\n";
    }
    std::cout << "}\n";
}
