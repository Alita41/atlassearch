#include "atlassearch/search_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <memory>
#include <queue>
#include <sstream>
#include <unordered_set>

namespace atlassearch {

struct Trie::Node {
    bool terminal = false;
    std::unordered_map<unsigned char, Node*> children;
};

Trie::Trie() : root_(new Node()) {}
Trie::~Trie() { destroy(root_); }

void Trie::destroy(Node* node) {
    if (!node) return;
    for (auto& [_, child] : node->children) destroy(child);
    delete node;
}

void Trie::insert(const std::string& token) {
    Node* cur = root_;
    for (unsigned char ch : token) {
        auto it = cur->children.find(ch);
        if (it == cur->children.end()) {
            Node* n = new Node();
            cur->children.emplace(ch, n);
            cur = n;
        } else {
            cur = it->second;
        }
    }
    cur->terminal = true;
}

void Trie::collect(const Node* node, std::string& current, std::vector<std::string>& out, std::size_t limit) const {
    if (!node || out.size() >= limit) return;
    if (node->terminal) out.push_back(current);
    if (out.size() >= limit) return;

    std::vector<unsigned char> keys;
    keys.reserve(node->children.size());
    for (const auto& [ch, _] : node->children) keys.push_back(ch);
    std::sort(keys.begin(), keys.end());
    for (unsigned char ch : keys) {
        current.push_back(static_cast<char>(ch));
        collect(node->children.at(ch), current, out, limit);
        current.pop_back();
        if (out.size() >= limit) break;
    }
}

std::vector<std::string> Trie::complete(const std::string& prefix, std::size_t limit) const {
    const Node* cur = root_;
    for (unsigned char ch : prefix) {
        auto it = cur->children.find(ch);
        if (it == cur->children.end()) return {};
        cur = it->second;
    }
    std::vector<std::string> out;
    std::string current = prefix;
    collect(cur, current, out, limit);
    return out;
}

SearchEngine::SearchEngine(double k1, double b) : k1_(k1), b_(b) {}

std::vector<std::string> SearchEngine::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    for (unsigned char ch : text) {
        // Input is expected to be normalized by Python. ASCII whitespace is enough
        // to separate UTF-8 Urdu/Roman-Urdu tokens without damaging bytes.
        if (std::isspace(ch)) {
            if (!cur.empty()) {
                tokens.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(static_cast<char>(ch));
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

void SearchEngine::add_document(std::uint32_t doc_id, const std::string& text) {
    auto tokens = tokenize(text);
    doc_lengths_[doc_id] = static_cast<std::uint32_t>(tokens.size());
    total_terms_ += tokens.size();

    std::unordered_map<std::string, std::uint32_t> tf;
    for (const auto& token : tokens) {
        ++tf[token];
        trie_.insert(token);
    }

    for (const auto& [term, count] : tf) {
        postings_[term].push_back({doc_id, count});
        ++document_frequency_[term];
    }
}

double SearchEngine::average_document_length() const {
    if (doc_lengths_.empty()) return 0.0;
    return static_cast<double>(total_terms_) / static_cast<double>(doc_lengths_.size());
}

std::vector<SearchResult> SearchEngine::search(const std::string& query, std::size_t top_k) const {
    if (doc_lengths_.empty() || top_k == 0) return {};
    const auto query_terms = tokenize(query);
    const double N = static_cast<double>(doc_lengths_.size());
    const double avgdl = std::max(average_document_length(), 1e-9);
    std::unordered_map<std::uint32_t, double> scores;

    for (const auto& term : query_terms) {
        auto p_it = postings_.find(term);
        if (p_it == postings_.end()) continue;
        const auto df_it = document_frequency_.find(term);
        if (df_it == document_frequency_.end()) continue;
        const double df = static_cast<double>(df_it->second);
        const double idf = std::log(1.0 + (N - df + 0.5) / (df + 0.5));

        for (const auto& posting : p_it->second) {
            const double tf = static_cast<double>(posting.tf);
            const double dl = static_cast<double>(doc_lengths_.at(posting.doc_id));
            const double denom = tf + k1_ * (1.0 - b_ + b_ * dl / avgdl);
            scores[posting.doc_id] += idf * (tf * (k1_ + 1.0)) / denom;
        }
    }

    struct HeapItem { double score; std::uint32_t doc_id; };
    auto cmp = [](const HeapItem& a, const HeapItem& b) { return a.score > b.score; };
    std::priority_queue<HeapItem, std::vector<HeapItem>, decltype(cmp)> heap(cmp);

    for (const auto& [doc_id, score] : scores) {
        if (heap.size() < top_k) heap.push({score, doc_id});
        else if (score > heap.top().score) {
            heap.pop();
            heap.push({score, doc_id});
        }
    }

    std::vector<SearchResult> out;
    out.reserve(heap.size());
    while (!heap.empty()) {
        out.push_back({heap.top().doc_id, heap.top().score});
        heap.pop();
    }
    std::sort(out.begin(), out.end(), [](const SearchResult& a, const SearchResult& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.doc_id < b.doc_id;
    });
    return out;
}

std::vector<std::string> SearchEngine::autocomplete(const std::string& prefix, std::size_t limit) const {
    return trie_.complete(prefix, limit);
}

}  // namespace atlassearch
