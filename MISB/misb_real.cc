// just do prefetch line. I'll just be in the . I tiwll send smething from the l2 to the last elvel for prefetch cache.
// #include "misb.h"

// Other objects defined in Champsim
#include "cache.h" // need this primarily so that I can work with the SP and the PS cache

#include "msl/lru_table.h"

// Includes for things not defined in Champsim
#include <cstdint>
#include <string>
#include <vector>

#include <unordered_map>

using namespace std;

// I need this for
uint64_t next_structural_address = 0;

// Assuming you have a map to store the physical to structural address mapping
std::unordered_map<uint64_t, uint64_t> physical_to_structural_address;

template <typename t>
class entry
{
public:
  bool _public;
  bool valid;
  bool dirty;   // check if the data is dirty and needs to be written back to the dram
  uint32_t lru; // this is the least recently used counter. Decide what to evict
  t data;       // this is the data that is stored in the cache. It can either be the physical or the structural address depending on the cache
  uint64_t
      physical_address; // This the address of the data that is stored in the cache. It can be the physical or the structural address depending on the cache
  uint64_t
      structural_address; // This the address of the data that is stored in the cache. It can be the physical or the structural address depending on the cache
  entry()
  {
    _public = false;
    dirty = false;
    lru = 0;
    data = 0;
    physical_address = 0;
    structural_address = 0;
    valid = false;
  }
  entry(uint64_t _physical_address, uint64_t _structural_address)
  {
    _public = false;
    dirty = false;
    lru = 0;
    data = 0;
    physical_address = _physical_address;
    structural_address = _structural_address;
    valid = false;
  }
};

template <typename t>
class cache_specialized
{
public:
  vector<entry<t>>& dram; // Reference to the shared dram vector
  vector<entry<t>> array; // The cache array
  uint32_t sets;
  uint32_t ways;
  uint32_t lru_count = 0;

  cache_specialized(uint32_t ways_, uint32_t sets_, vector<entry<t>>& dram_) : dram(dram_), sets(sets_), ways(ways_)
  {
    // No need to initialize dram here, as it's passed as a reference
    array.resize(sets * ways);
  }

  uint64_t read(uint64_t address, uint32_t sets, uint32_t ways, cache_specialized<uint64_t>& other_cache, bool is_PS, uint64_t structural_address);
  void write(entry<t> data, uint64_t address);
};

template <typename t>
uint64_t cache_specialized<t>::read(uint64_t address, uint32_t sets, uint32_t ways, cache_specialized<uint64_t>& other_cache, bool is_PS,
                                    uint64_t structural_address)
{
  uint32_t min_lru = lru_count + 10;
  lru_count++;
  uint32_t set_index = (address / ways) % sets;
  uint32_t min_lru_way = ways;
  int found = -1;
  uint64_t addressToPrefetch = 0;

  for (int i = 0; i < ways; i++) {
     size_t index = (set_index * ways) + i;
    if (index >= array.size()) {
      // Handle the error, for example by returning a special value
      
      return address + (1 << LOG2_BLOCK_SIZE);  
    }
    if (array.at((set_index * ways) + i).physical_address == address) { // if the address is in the cache
      array.at(set_index * ways + i).lru = lru_count;
      
      if (is_PS) { // checking to see if the PS cache is calling this and also want to make sure that it has the structural address.
        // If the PS request hits, MISB predicts prefetch requests for the next few structural addresses.
        // If the PS table does not contain the address then I will add it to the PS cache
        uint64_t next_structural_address = structural_address + 1;
        if (physical_to_structural_address.find(next_structural_address) != physical_to_structural_address.end()) {
          // If the next structural address is in the map, prefetch it
          addressToPrefetch = physical_to_structural_address[next_structural_address];
          entry<t> dataToWrite;
          dataToWrite.physical_address = addressToPrefetch;
          dataToWrite.structural_address = next_structural_address; // Assuming you have the next structural address available
          // Set other fields of dataToWrite as needed

          this->write(dataToWrite, addressToPrefetch);
        }
      }

    } else { // If the PS request misses, MISB issues an off-chip PS load request, delaying the prediction until the request completes.
      // When the request completes, the new mappings are inserted into both the PS and SP caches
      entry<uint64_t> newEntry(address, structural_address);
      this->write(newEntry, address);
      other_cache.write(newEntry, address);

      addressToPrefetch = dram.at(address).physical_address;
    }

    if (array.at(set_index).lru < min_lru) {
      // handle the eviction
      min_lru = array.at(set_index).lru;
      min_lru_way = i;
    }
    // Regardless of whether the PS load hits or misses in the cache, when we find its structural address s
    // we issue a data prefetch request for structural address s + 1
  }
  return addressToPrefetch;
}
template <typename t>
void cache_specialized<t>::write(entry<t> data, uint64_t addr)
{
  uint32_t min_lru_way = 0;
  uint32_t min_lru = lru_count++;
  uint32_t set_index = (addr / ways) % sets;

  for (int i = 0; i < ways; i++) {
    if (array.at((set_index * ways) + i).physical_address == addr && array.at((set_index * ways) + i).valid) {
      array.at((set_index * ways) + i) = data;
      array.at((set_index * ways) + i).dirty = true;
      array.at((set_index * ways) + i).lru = lru_count;
      return;
    }

    if (array.at((set_index * ways) + i).lru < min_lru) {
      min_lru = array.at((set_index * ways) + i).lru;
      min_lru_way = i;
    }
  }

  // Evict the least recently used entry and write new data
  array.at((set_index * ways) + min_lru_way).valid = false;
  if (array.at((set_index * ways) + min_lru_way).dirty) {
    // If the evicted entry is dirty, write it back to the dram
    dram.at((set_index * ways) + min_lru_way) = array.at((set_index * ways) + min_lru_way);
  }
  array.at((set_index * ways) + min_lru_way) = data;
  array.at((set_index * ways) + min_lru_way).valid = true;
  array.at((set_index * ways) + min_lru_way).dirty = false;
  array.at((set_index * ways) + min_lru_way).lru = lru_count;
}

uint64_t address = 0;
bool cacheMiss = false;

struct BloomFilter {
  std::vector<bool> set;
  int size;

  BloomFilter() : size(17 * 1024 * 8), set(size, false) {}

  void add(int item)
  { // adding the item
    for (int i = 0; i < 2; ++i) {
      set[hash(item, i)] = true;
    }
  }

  bool contains(int item) const
  { // checking if the item is in the bloom filter
    for (int i = 0; i < 2; ++i) {
      if (!set[hash(item, i)]) {
        return false;
      }
    }
    return true;
  }

  int hash(int item, int i) const
  {
    // This is a h3 hash. it is what they use in the paper for the bloom filter.
    int a = i;
    int b = i + 1;
    int p = 104792; // have to pick a random large prime number

    return ((a * item + b) % p) % size;
  }
};

BloomFilter bloom_filter;

// Constants
static constexpr uint32_t dramSize =  147483647; // Adjust as needed
static constexpr uint32_t NumSets = 128;   // Adjust as needed
static constexpr uint32_t NumWays = 8;     // Adjust as needed

vector<entry<uint64_t>> dram(dramSize);

// Initialize the dram array
// dram.resize(dramSize);

// Global variable declaration

std::unordered_map<uint64_t, uint64_t> pc_to_structural_address;

// Global variables
cache_specialized<uint64_t> PS_cache(NumSets, NumWays, dram);
cache_specialized<uint64_t> SP_cache(NumSets, NumWays, dram);

uint64_t get_structural_address(uint64_t ip, uint64_t physical_address); // Function declaration

/***********************************************************************************************************************************************/
//*************************Working with Champsim**************************************/
/***********************************************************************************************************************************************/

// This function is called when the cache is initialized. You can use it to initialize elements of dynamic structures, such as std::vector or std::map.
void CACHE::prefetcher_initialize()
{
  for (int i = 0; i < dramSize; i++) {
    dram[i].physical_address = i;
  }
}
uint64_t misb_prefetch(uint64_t addr, uint64_t ip)
{
    if(addr >= dramSize ){
        return addr + (1 << LOG2_BLOCK_SIZE);
    }

    uint64_t addressToPrefetch = PS_cache.read(addr, NumSets, NumWays, SP_cache, true, get_structural_address(ip, addr));
    return (addressToPrefetch);
}


uint32_t CACHE::prefetcher_cache_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, bool useful_prefetch, uint8_t type, uint32_t metadata_in)
{
    uint64_t addressToPrefetch = 0;
  if (!cache_hit) {          // if there is a miss in the cache
     addressToPrefetch = misb_prefetch(addr, ip); // I want to prefetch that line
  }
     prefetch_line(addressToPrefetch, true, 0);
  return metadata_in;
}


uint32_t CACHE::prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in)
{
  return metadata_in;
}

// I need to create a ps and Sp stable. there I will check for the values

void CACHE::prefetcher_cycle_operate() {}

void CACHE::prefetcher_final_stats() {}

/***********************************************************************************************************************************************/
//**************Helper Functions**************************************************/
/***********************************************************************************************************************************************/

bool isFull_cache(cache_specialized<uint64_t> cache, uint32_t set_index)
{
  for (int i = 0; i < cache.ways; i++) {
    if (!cache.array.at(set_index * cache.ways + i).valid) {
      return false;
    }
  }
  return true;
}

void evict_cache(cache_specialized<uint64_t>& cache, uint32_t set_index)
{
  if (isFull_cache(cache, set_index)) {
    int min_lru = INT_MAX;
    int min_lru_way = -1;
    for (int i = 0; i < cache.ways; i++) {
      if (cache.array.at(set_index * cache.ways + i).lru < min_lru) {
        min_lru = cache.array.at(set_index * cache.ways + i).lru;
        min_lru_way = i;
      }
    }
    if (min_lru_way != -1) {
      cache.array.at(set_index * cache.ways + min_lru_way).valid = false;
    }
  }
}

// Function to get the structural address for a given PC and physical address
uint64_t get_structural_address(uint64_t ip, uint64_t physical_address)
{

  // Check if the PC is already in the map
  if (pc_to_structural_address.find(ip) == pc_to_structural_address.end()) {
    // If it's not in the map, assign the next structural address to it
    pc_to_structural_address[ip] = next_structural_address++;
  }

  // Map the physical address to the structural address
  physical_to_structural_address[physical_address] = pc_to_structural_address[ip];

  // Return the structural address for this PC
  return pc_to_structural_address[ip] % 1024;
}