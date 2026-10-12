#include "atlassearch/search_engine.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using atlassearch::SearchEngine;

static bool load_tsv(const std::string& path, SearchEngine& engine) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    std::size_t loaded = 0;
    while (std::getline(in, line)) {
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        const auto id = static_cast<std::uint32_t>(std::stoul(line.substr(0, tab)));
        const auto text = line.substr(tab + 1);
        engine.add_document(id, text);
        ++loaded;
    }
    std::cerr << "READY\t" << loaded << "\t" << engine.vocabulary_size() << "\n";
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: atlassearch_bm25 <normalized_docs.tsv>\n";
        return 2;
    }

    SearchEngine engine;
    if (!load_tsv(argv[1], engine)) {
        std::cerr << "failed to load: " << argv[1] << "\n";
        return 3;
    }

    // Persistent line protocol used by the Python API:
    // S\t<k>\t<query>  -> R\t<doc_id>:<score>,...
    // A\t<k>\t<prefix> -> C\t<token>|<token>|...
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT") break;
        std::istringstream ss(line);
        std::string op, k_s, payload;
        std::getline(ss, op, '\t');
        std::getline(ss, k_s, '\t');
        std::getline(ss, payload);
        std::size_t k = 10;
        try { k = static_cast<std::size_t>(std::stoul(k_s)); } catch (...) {}

        if (op == "S") {
            auto results = engine.search(payload, k);
            std::cout << "R\t";
            for (std::size_t i = 0; i < results.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << results[i].doc_id << ':' << std::fixed << std::setprecision(8) << results[i].score;
            }
            std::cout << '\n' << std::flush;
        } else if (op == "A") {
            auto results = engine.autocomplete(payload, k);
            std::cout << "C\t";
            for (std::size_t i = 0; i < results.size(); ++i) {
                if (i) std::cout << '|';
                std::cout << results[i];
            }
            std::cout << '\n' << std::flush;
        } else {
            std::cout << "E\tunknown_op\n" << std::flush;
        }
    }
    return 0;
}
