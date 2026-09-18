//  src/python_bindings/fastvdf.cpp
//  Original bindings + ContinuousVDF (minimal modification)
//  ---------------------------------------------------------------------------
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gmp.h>

#include "../verifier.h"
#include "../prover_slow.h"
#include "../alloc.hpp"
#include "../proof_common.h"

namespace py = pybind11;

// ═════════════════════════════════════════════════════════════════════════════
//  Original bindings (unchanged)
// ═════════════════════════════════════════════════════════════════════════════
PYBIND11_MODULE(chiavdf, m) {
    m.doc() = "Chia proof of time";

    // Creates discriminant.
    m.def("create_discriminant", [] (const py::bytes& challenge_hash, int discriminant_size_bits) {
        std::string challenge_hash_str(challenge_hash);
        integer D;
        {
            py::gil_scoped_release release;
            auto challenge_hash_bits = std::vector<uint8_t>(challenge_hash_str.begin(), challenge_hash_str.end());
            D = CreateDiscriminant(
                challenge_hash_bits,
                discriminant_size_bits
            );
        }
        return D.to_string();
    });

    // Checks a simple wesolowski proof.
    m.def("verify_wesolowski", [] (const std::string& discriminant,
                                   const std::string& x_s, const std::string& y_s,
                                   const std::string& proof_s,
                                   uint64_t num_iterations) {
        integer D(discriminant);
        std::string x_s_copy(x_s);
        std::string y_s_copy(y_s);
        std::string proof_s_copy(proof_s);
        bool is_valid = false;
        {
            py::gil_scoped_release release;
            form x = DeserializeForm(D, (const uint8_t *)x_s_copy.data(), x_s_copy.size());
            form y = DeserializeForm(D, (const uint8_t *)y_s_copy.data(), y_s_copy.size());
            form proof = DeserializeForm(D, (const uint8_t *)proof_s_copy.data(), proof_s_copy.size());
            VerifyWesolowskiProof(D, x, y, proof, num_iterations, is_valid);
        }
        return is_valid;
    });

    // Checks an N wesolowski proof.
    m.def("verify_n_wesolowski", [] (const std::string& discriminant,
                                   const std::string& x_s,
                                   const std::string& proof_blob,
                                   const uint64_t num_iterations, const uint64_t disc_size_bits, const uint64_t recursion) {
        std::string discriminant_copy(discriminant);
        std::string x_s_copy(x_s);
        std::string proof_blob_copy(proof_blob);
        uint8_t *proof_blob_ptr = reinterpret_cast<uint8_t *>(proof_blob_copy.data());
        int proof_blob_size = proof_blob_copy.size();
        bool is_valid = false;
        {
            py::gil_scoped_release release;
            is_valid=CheckProofOfTimeNWesolowski(integer(discriminant_copy), (const uint8_t *)x_s_copy.data(), proof_blob_ptr, proof_blob_size, num_iterations, disc_size_bits, recursion);
        }
        return is_valid;
    });

    // Checks an N wesolowski proof.
    m.def("create_discriminant_and_verify_n_wesolowski", [] (const py::bytes& challenge_hash,
                                   const int discriminant_size_bits,
                                   const std::string& x_s,
                                   const std::string& proof_blob,
                                   const uint64_t num_iterations,
                                   const uint64_t recursion) {
        std::string challenge_hash_str(challenge_hash);
        std::vector<uint8_t> challenge_hash_bits = std::vector<uint8_t>(challenge_hash_str.begin(), challenge_hash_str.end());
        std::string x_s_copy(x_s);
        std::string proof_blob_copy(proof_blob);
        bool is_valid = false;
        {
            py::gil_scoped_release release;
            is_valid=CreateDiscriminantAndCheckProofOfTimeNWesolowski(challenge_hash_bits, discriminant_size_bits,(const uint8_t *)x_s_copy.data(), (const uint8_t *)proof_blob_copy.data(), proof_blob_copy.size(), num_iterations, recursion);
        }
        return is_valid;
    });

    m.def("prove", [] (const py::bytes& challenge_hash, const std::string& x_s, int discriminant_size_bits, uint64_t num_iterations, const std::string& shutdown_file_path) {
        std::string challenge_hash_str(challenge_hash);
        std::string x_s_copy(x_s);
        std::vector<uint8_t> result;
        std::string shutdown_file_path_copy(shutdown_file_path);
        {
            py::gil_scoped_release release;
            std::vector<uint8_t> challenge_hash_bytes(challenge_hash_str.begin(), challenge_hash_str.end());
            integer D = CreateDiscriminant(
                    challenge_hash_bytes,
                    discriminant_size_bits
            );
            form x = DeserializeForm(D, (const uint8_t *) x_s_copy.data(), x_s_copy.size());
            result = ProveSlow(D, x, num_iterations, shutdown_file_path_copy);
        }
        py::bytes ret = py::bytes(reinterpret_cast<char*>(result.data()), result.size());
        return ret;
    });

    // Checks an N wesolowski proof, given y is given by 'GetB()' instead of a form.
    m.def("verify_n_wesolowski_with_b", [] (const std::string& discriminant,
                                   const std::string& B,
                                   const std::string& x_s,
                                   const std::string& proof_blob,
                                   const uint64_t num_iterations, const uint64_t recursion) {
        std::string discriminant_copy(discriminant);
        std::string B_copy(B);
        std::string x_s_copy(x_s);
        std::string proof_blob_copy(proof_blob);
        std::pair<bool, std::vector<uint8_t>> result;
        {
            py::gil_scoped_release release;
            uint8_t *proof_blob_ptr = reinterpret_cast<uint8_t *>(proof_blob_copy.data());
            int proof_blob_size = proof_blob_copy.size();
            result = CheckProofOfTimeNWesolowskiWithB(integer(discriminant_copy), integer(B_copy), (const uint8_t *)x_s_copy.data(), proof_blob_ptr, proof_blob_size, num_iterations, recursion);
        }
        py::bytes res_bytes = py::bytes(reinterpret_cast<char*>(result.second.data()), result.second.size());
        py::tuple res_tuple = py::make_tuple(result.first, res_bytes);
        return res_tuple;
    });

    m.def("get_b_from_n_wesolowski", [] (const std::string& discriminant,
                                   const std::string& x_s,
                                   const std::string& proof_blob,
                                   const uint64_t num_iterations, const uint64_t recursion) {
        std::string discriminant_copy(discriminant);
        std::string x_s_copy(x_s);
        std::string proof_blob_copy(proof_blob);
        integer B;
        {
            py::gil_scoped_release release;
            uint8_t *proof_blob_ptr = reinterpret_cast<uint8_t *>(proof_blob_copy.data());
            int proof_blob_size = proof_blob_copy.size();
            B = GetBFromProof(integer(discriminant_copy), (const uint8_t *)x_s_copy.data(), proof_blob_ptr, proof_blob_size, num_iterations, recursion);
        }
        return B.to_string();
    });

    m.def("prove_from_hash", [](
            const py::bytes& challenge_hash,
            int discriminant_size_bits = 2048,
            uint64_t num_iterations = 10000
        ) {
        std::string hstr(challenge_hash);
        if (hstr.size() != 32) {
            throw std::runtime_error("challenge_hash must be exactly 32 bytes");
        }
        std::vector<uint8_t> hash_bytes(hstr.begin(), hstr.end());

        std::vector<uint8_t> proof_blob;
        {
            py::gil_scoped_release _unlock;
            
            // Debug point 1: Before creating discriminant
            // std::cerr << "=== prove_from_hash DEBUG ===" << std::endl;
            // std::cerr << "discriminant_size_bits: " << discriminant_size_bits << std::endl;
            
            // 2) Build the discriminant
            integer D = CreateDiscriminant(hash_bytes, discriminant_size_bits);
            
            // Debug point 2: After creating discriminant
            // std::cerr << "D created successfully" << std::endl;
            
            // Check the modulus - need to create integer objects for 8 and 4
            integer eight(8);
            integer four(4);
            integer two(2);
            integer one(1);
            integer D_mod_8 = D % eight;
            integer D_mod_4 = D % four;
            
            // Convert to long for printing (assuming they fit)
            // std::cerr << "D % 8 = " << mpz_get_si(D_mod_8.impl) << std::endl;
            // std::cerr << "D % 4 = " << mpz_get_si(D_mod_4.impl) << std::endl;
            // std::cerr << "D < 0 ? " << (mpz_sgn(D.impl) < 0 ? "yes" : "no") << std::endl;
            
            // Debug point 3: Before creating form
            // std::cerr << "About to create form (two, one, D)" << std::endl;
            
            try {
                // 3) Map hash → form
                form x = form::from_abd(two, one, D);
                
                // std::cerr << "Form created successfully" << std::endl;
                
                // Debug point 4: Before reduce
                // std::cerr << "About to reduce form" << std::endl;
                
                x.reduce();
                
                // std::cerr << "Form reduced successfully" << std::endl;
                
                // 4) Run the slow prover
                // std::cerr << "Starting ProveSlow with " << num_iterations << " iterations" << std::endl;
                proof_blob = ProveSlow(D, x, num_iterations, /*shutdown=*/"");
                
                // std::cerr << "Proof complete, size: " << proof_blob.size() << " bytes" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "ERROR: Exception caught: " << e.what() << std::endl;
                throw;
            } catch (...) {
                std::cerr << "ERROR: Unknown exception caught" << std::endl;
                throw;
            }
        }

        return py::bytes(reinterpret_cast<char*>(proof_blob.data()), proof_blob.size());
    });

    // py::class_<StreamingProver>(m, "StreamingProver")
    //     .def(py::init([](py::bytes challenge_hash,
    //                     int  discr_bits,
    //                     uint64_t N)
    //         {
    //             std::string s(challenge_hash);                // 32 bytes
    //             std::vector<uint8_t> v(s.begin(), s.end());
    //             return new StreamingProver(v, discr_bits, N);
    //         }),
    //         py::arg("challenge_hash"),
    //         py::arg("discriminant_size_bits") = 1024,
    //         py::arg("N") = 10'000)
    //     .def("next",
    //          [](StreamingProver& self,
    //             const std::string& shutdown_file)
    //          {
    //              auto v = self.next_raw(shutdown_file);
    //              return py::bytes(reinterpret_cast<char*>(v.data()), v.size());
    //          },
    //          py::arg("shutdown_file") = "")
    //     .def_property_readonly("total_iterations",
    //         &StreamingProver::total_iterations)
    //     .def_property_readonly("D",
    //         [](const StreamingProver& self) {
    //             return self.discriminant().to_vector();
    //         });

    py::class_<ThreadedStreamingProver>(m, "StreamingProver")
      .def(py::init([](py::bytes challenge_hash,
                       int       discr_bits,
                       uint64_t  checkpoint_n,
                       uint64_t  max_iters,
                       uint64_t  proof_interval_ms)
           {
               // Convert Python bytes -> std::vector<uint8_t>
               std::string s = challenge_hash;  // must be exactly 32 bytes
               if (s.size() != 32)
                   throw std::runtime_error("challenge_hash must be exactly 32 bytes");
               std::vector<uint8_t> v(s.begin(), s.end());
               return new ThreadedStreamingProver(
                   std::move(v),
                   discr_bits,
                   checkpoint_n,
                   max_iters,
                   proof_interval_ms
               );
           }),
           py::arg("challenge_hash"),
           py::arg("discriminant_size_bits") = 1024,
           py::arg("checkpoint_n")           = 10'000,
           py::arg("max_iters")              = 100'000'000,
           py::arg("proof_interval_ms")      = 1'000,
           "Create a streaming prover.  \n\n"
           "  • challenge_hash: 32-byte SHA-256 digest  \n"
           "  • discriminant_size_bits: size of the class group  \n"
           "  • checkpoint_n: number of squarings per proof chunk  \n"
           "  • max_iters: maximum total squarings supported  \n"
           "  • proof_interval_ms: (ignored—proofs fire immediately each chunk)")
        .def("start", &ThreadedStreamingProver::start,
             "Start the prover threads. Must be called before using the prover.")
        .def("get_last_available_proof",
            [](ThreadedStreamingProver &self) {
                auto pr = self.get_last_available_proof();
                const auto &blob = pr.first;
                uint64_t iters   = pr.second;
                // return (bytes, int)
                return std::make_pair(
                    py::bytes(reinterpret_cast<const char*>(blob.data()), blob.size()),
                    iters
                );
            },
            "Get the last proof as (blob: bytes, iterations: int); empty blob if none yet.")
        .def("get_current_iterations", &ThreadedStreamingProver::get_current_iterations,
             "Get the current number of iterations completed")
        .def("set_verbose", &ThreadedStreamingProver::set_verbose,
             "Enable/disable verbose logging")
        .def("stop", &ThreadedStreamingProver::stop,
             "Stop the prover threads")
        .def("reset", [](ThreadedStreamingProver &self, py::bytes new_challenge_hash) {
            std::string s = new_challenge_hash;
            if (s.size() != 32)
                throw std::runtime_error("challenge_hash must be exactly 32 bytes");
            std::vector<uint8_t> v(s.begin(), s.end());
            self.reset(std::move(v));
        }, py::arg("new_challenge_hash"),
        "Reset the prover with a new challenge hash");  // <- This semicolon ends the whole statement

    m.def("verify_from_hash", [](
            const py::bytes& challenge_hash,
            const py::bytes& proof_blob,
            int discriminant_size_bits = 2048,
            uint64_t num_iterations    = 10000,
            uint64_t recursion         = 0
        ) {
        // 1) Grab & check
        std::string hstr(challenge_hash);
        if (hstr.size() != 32) {
            throw std::runtime_error("challenge_hash must be exactly 32 bytes");
        }
        std::vector<uint8_t> hash_bytes(hstr.begin(), hstr.end());
        std::string proof_str(proof_blob);
        const uint8_t* proof_ptr = reinterpret_cast<const uint8_t*>(proof_str.data());
        int proof_len = int(proof_str.size());

        bool ok = false;
        {
            py::gil_scoped_release _unlock;

            // 2) Rebuild discriminant
            integer D = CreateDiscriminant(hash_bytes, discriminant_size_bits);
            integer two(2);
            integer one(1);

            // 3) Same hash→form mapping
            form x = form::from_abd(two, one, D);
            x.reduce();

            // 4) Serialize & verify
            int real_bits = int(D.num_bits());
            std::vector<uint8_t> x_bytes = SerializeForm(x, real_bits);
            ok = CheckProofOfTimeNWesolowski(
                D,
                x_bytes.data(),    // initial form
                proof_ptr, proof_len,
                num_iterations,
                real_bits,
                recursion
            );
        }
        return ok;
    });


}