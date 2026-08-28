#include "formats.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int n, char ** v) {
    try {
        if (n < 3) throw std::runtime_error("usage: ramgpt-compare-reference A.rql B.rql [positions.jsonl] [--strict]");
        bool strict = false;
        std::string output;
        for (int i = 3; i < n; ++i) {
            if (std::string(v[i]) == "--strict") strict = true;
            else if (output.empty()) output = v[i];
            else throw std::runtime_error("unexpected argument");
        }
        std::ofstream f;
        if (!output.empty()) f.open(output);
        auto c = rq::compare(v[1], v[2], output.empty() ? nullptr : &f,
                             strict ? rq::CompatibilityMode::Strict : rq::CompatibilityMode::Basic);
        std::cout << std::setprecision(12)
                  << "number_of_positions=" << c.positions
                  << "\nsame_top1_rate=" << (c.positions ? double(c.positions-c.changed_top1)/c.positions : 0)
                  << "\nsame_top2_rate=" << (c.positions ? double(c.positions-c.changed_top2)/c.positions : 0)
                  << "\nchanged_top1_count=" << c.changed_top1
                  << "\nmaximum_absolute_logit_difference=" << c.max_abs
                  << "\nmean_absolute_logit_difference=" << (c.values ? c.sum_abs/c.values : 0)
                  << "\nRMS_logit_difference=" << (c.values ? std::sqrt(c.sum_sq/c.values) : 0)
                  << "\ndiffering_float32_values=" << c.differing_values
                  << "\nfirst_differing_position=";
        if (c.differing_values) std::cout << c.first_differing_position; else std::cout << "none";
        std::cout << "\nfirst_differing_vocab_id=";
        if (c.differing_values) std::cout << c.first_differing_vocab_id; else std::cout << "none";
        std::cout << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
