#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace atlassearch {

struct Posting {
    std::uint32_t doc_id;
    std::uint32_t tf;
};

struct SearchResult {
    std::uint32_t doc_id;
    double score;
};

class Trie {
public:
    Trie();
    ~Trie();
    Trie(const Trie&) = delete;
    Trie& operator=(const Trie&) = delete;

    void insert(const std::string& token);
    std::vector<std::string> complete(const std::string& prefix, std::size_t limit) const;

private:
    struct Node;
    Node* root_;
    void destroy(Node* node);
    void collect(const Node* node, std::string& current, std::vector<std::string>& out, std::size_t limit) const;
};

class SearchEngine {
public:
    explicit SearchEngine(double k1 = 1.2, double b = 0.75);

    void add_document(std::uint32_t doc_id, const std::string& text);
    std::vector<SearchResult> search(const std::string& query, std::size_t top_k) const;
    std::vector<std::string> autocomplete(const std::string& prefix, std::size_t limit) const;

    std::size_t document_count() const { return doc_lengths_.size(); }
    std::size_t vocabulary_size() const { return postings_.size(); }
    double average_document_length() const;

    static std::vector<std::string> tokenize(const std::string& text);

private:
    double k1_;
    double b_;
    std::unordered_map<std::string, std::vector<Posting>> postings_;
    std::unordered_map<std::string, std::uint32_t> document_frequency_;
    std::unordered_map<std::uint32_t, std::uint32_t> doc_lengths_;
    std::uint64_t total_terms_ = 0;
    Trie trie_;
};

}  // namespace atlassearch
