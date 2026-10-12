#include "atlassearch/search_engine.hpp"
#include <cassert>
#include <iostream>

int main() {
    atlassearch::SearchEngine engine;
    engine.add_document(1, "machine learning search engine");
    engine.add_document(2, "database systems and indexing");
    engine.add_document(3, "machine learning retrieval ranking");

    auto r = engine.search("machine learning", 2);
    assert(r.size() == 2);
    assert(r[0].doc_id == 1 || r[0].doc_id == 3);

    auto c = engine.autocomplete("mach", 5);
    assert(!c.empty());
    assert(c[0] == "machine");
    std::cout << "ok\n";
    return 0;
}
