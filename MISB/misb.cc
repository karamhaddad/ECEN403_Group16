#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <deque>
#include <memory> // Include smart pointer header
#include "cache.h"
#include "msl/lru_table.h"

// Entry Class for physical and structural addresses
//This is required to make keeping the strucutal address for each physical together
class Entry {
public:
    bool valid;
    bool dirty;
    uint32_t lru;
    uint64_t physical_address;
    uint64_t structural_address;

    Entry() : valid(false), dirty(false), lru(0), physical_address(0), structural_address(0) {}
    Entry(uint64_t _physical_address, uint64_t _structural_address)
        : valid(true), dirty(false), lru(0), physical_address(_physical_address), structural_address(_structural_address) {}
};

// BloomFilter 
//needs to use a hash and is checked before going off chip
class BloomFilter {
public:
    int size;
    std::vector<bool> set;

    BloomFilter() : size(17 * 1024 * 8), set(size, false) {}

    void add(int item) {
        for (int i = 0; i < 2; ++i) {
            int index = hash(item, i);
            if (index >= 0 && index < size) {
                set[index] = true;
            } else {
               // std::cerr << "[ERROR] BloomFilter invalid index: " << index << "\n";
            }
        }
    }

    bool contains(int item) const {
        for (int i = 0; i < 2; ++i) {
            int index = hash(item, i);
            if (index < 0 || index >= size || !set[index]) {
                return false;
            }
        }
        return true;
    }

private:
    int hash(int item, int i) const {
        int a = i;
        int b = i + 1;
        int p = 104792; // prime number for hashing
        int result = ((a * item + b) % p) % size;
        if (result < 0 || result >= size) {
            //std::cerr << "[ERROR] BloomFilter hash out-of-bounds: result=" << result << ", size=" << size << "\n";
        }
        return result;
    }
};

//needed to make a seperate cache structure to implement the ps and the sp cache. 
template <typename T>
class SpecializedCache {
public:
    std::vector<Entry> array;
    uint32_t sets;
    uint32_t ways;
    uint32_t lru_count;

    SpecializedCache(uint32_t sets_, uint32_t ways_) : sets(sets_), ways(ways_), lru_count(0) {
        array.resize(sets * ways);
        for (auto& entry : array) {
            entry = Entry(); 
        }
    }

    uint64_t read(T address, bool is_ps_cache) {
        uint32_t set_index = (address / ways) % sets;
        for (uint32_t i = 0; i < ways; ++i) {
            uint32_t index = set_index * ways + i;
            if (index >= array.size()) {
                //std::cerr << "[ERROR] SpecializedCache out-of-bounds read: index=" << index << ", size=" << array.size() << "\n";
                return -1; // Cache miss
            }
            if (array[index].valid) {
                if ((is_ps_cache && array[index].physical_address == address) ||
                    (!is_ps_cache && array[index].structural_address == address)) {
                    array[index].lru = lru_count++;
                    return is_ps_cache ? array[index].structural_address : array[index].physical_address;
                }
            }
        }
        return -1; // Cache miss
    }

    void write(Entry data, T address) {
        uint32_t set_index = (address / ways) % sets;
        for (uint32_t i = 0; i < ways; ++i) {
            uint32_t index = set_index * ways + i;
            if (index >= array.size()) {
                //std::cerr << "[ERROR] SpecializedCache out-of-bounds write: index=" << index << ", size=" << array.size() << "\n";
                return;
            }
            if (!array[index].valid) {
                array[index] = data;
                array[index].lru = lru_count++;
                return;
            }
        }
        evict_entry(set_index);
        write(data, address);
    }

    void evict_entry(uint32_t set_index) {
        uint32_t min_lru = lru_count + 10;
        int evict_way = -1;
        for (uint32_t i = 0; i < ways; ++i) {
            uint32_t index = set_index * ways + i;
            if (index >= array.size()) {
                //std::cerr << "[ERROR] SpecializedCache out-of-bounds evict: index=" << index << ", size=" << array.size() << "\n";
                return;
            }
            if (array[index].lru < min_lru) {
                min_lru = array[index].lru;
                evict_way = i;
            }
        }
        if (evict_way != -1) {
            array[set_index * ways + evict_way].valid = false;
        }
    }
};

// Global Data Structures
std::unique_ptr<BloomFilter> bloom_filter = std::make_unique<BloomFilter>();
std::unique_ptr<SpecializedCache<uint64_t>> specialized_ps_cache = std::make_unique<SpecializedCache<uint64_t>>(128, 8);
std::unique_ptr<SpecializedCache<uint64_t>> specialized_sp_cache = std::make_unique<SpecializedCache<uint64_t>>(128, 8);
std::unordered_map<uint64_t, std::vector<int64_t>> ip_stride_map; // This is being used for the training unit so that it can detect patterns well
std::deque<uint64_t> recent_structural_buffer; // recent structural addresses

// Prefetching Statistics coutners to see if every part of the prefetcher is working correctly
uint32_t total_prefetches = 0;
uint32_t ps_cache_hits = 0;
uint32_t ps_cache_misses = 0;
uint32_t sp_cache_hits = 0;
uint32_t sp_cache_misses = 0;
uint32_t bloom_filter_hits = 0;
uint32_t bloom_filter_misses = 0;

// Prefetch Logic for Structural Addresses
void prefetch_structural_addresses(uint64_t base_structural_address, uint32_t metadata_in, CACHE* cache) {
    for (int i = 1; i <= 3; ++i) {
        uint64_t next_structural_address = base_structural_address + i;
        if (next_structural_address > UINT64_MAX / 2) {
            std::cerr << "[ERROR] Prefetching invalid structural address: " << next_structural_address << "\n";
        }

        uint64_t next_physical_address = specialized_sp_cache->read(next_structural_address, false);
        if (next_physical_address == (uint64_t)-1 || next_physical_address == 0) {
            ++sp_cache_misses;

            specialized_sp_cache->write(Entry(0, next_structural_address), next_structural_address);
        }
    }
}

// Prefetcher Cache Operate Function
// This function has most of the logic and uses the address and the ip given from the function
// It tracks prefetching statistics to make sure that every part of the prefetcher works as intended
uint32_t CACHE::prefetcher_cache_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, bool useful_prefetch, uint8_t type, uint32_t metadata_in) {
    ++total_prefetches;  // Increment the total prefetch count for statistics.

    // error checking to make sure that the address is valid
    if (addr == 0 || addr > UINT64_MAX / 2) {
        std::cerr << "[ERROR] Invalid physical address: " << addr << "\n";
        return metadata_in;  // Return the unchanged metadata.
    }

    // try to get the structual address.
    uint64_t structural_address = specialized_ps_cache->read(addr, true);

    if (structural_address == (uint64_t)-1) {
        ++ps_cache_misses;  // Increment PS cache miss count.

        // Check if the is in the Bloom filter.
        if (!bloom_filter->contains(addr)) {
            ++bloom_filter_misses;  // Increment Bloom filter miss count.

            // Generate a new structural address and add it to the PS and SP caches, and the Bloom filter.
            uint64_t new_structural_address = addr / BLOCK_SIZE;
            specialized_ps_cache->write(Entry(addr, new_structural_address), addr);  // Add to PS cache.
            specialized_sp_cache->write(Entry(addr, new_structural_address), new_structural_address);  // Add to SP cache.
            bloom_filter->add(addr);  // add to the Bloom filter.
            structural_address = new_structural_address;  // Update the structural address.
        } else {
            ++bloom_filter_hits;  // Increment Bloom filter hit count.
        }
    } else {
        ++ps_cache_hits;  // Increment PS cache hit count.
    }

    // If a valid structural address is found or generated, initiate prefetching.
    if (structural_address != (uint64_t)-1) {
        prefetch_structural_addresses(structural_address, metadata_in, this);  // Issue prefetch requests.
    }

    // Return the metadata, which may be used for further processing in the pipeline.
    return metadata_in;
}



// Prefetcher Final Stats
void CACHE::prefetcher_final_stats() {
    //probably not needed but did just in case so that there was no issue
    // Output current stats
    std::cout << "----- Prefetching Statistics -----\n";
    std::cout << "Total Prefetches: " << total_prefetches << "\n";
    std::cout << "PS Cache Hits: " << ps_cache_hits << "\n";
    std::cout << "PS Cache Misses: " << ps_cache_misses << "\n";
    std::cout << "SP Cache Hits: " << sp_cache_hits << "\n";
    std::cout << "SP Cache Misses: " << sp_cache_misses << "\n";
    std::cout << "Bloom Filter Hits: " << bloom_filter_hits << "\n";
    std::cout << "Bloom Filter Misses: " << bloom_filter_misses << "\n";

    // Reset all variables
    total_prefetches = 0;
    ps_cache_hits = 0;
    ps_cache_misses = 0;
    sp_cache_hits = 0;
    sp_cache_misses = 0;
    bloom_filter_hits = 0;
    bloom_filter_misses = 0;


    // Reset specialized caches
    specialized_ps_cache = std::make_unique<SpecializedCache<uint64_t>>(128, 8);
    specialized_sp_cache = std::make_unique<SpecializedCache<uint64_t>>(128, 8);

    // Reset Bloom filter
    bloom_filter = std::make_unique<BloomFilter>();

    std::cout << "Data structures reset for the next simulation.\n";
}

// Other Cache Methods
void CACHE::prefetcher_initialize() {}
uint32_t CACHE::prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in) {
    return metadata_in;
}
void CACHE::prefetcher_cycle_operate() {}
