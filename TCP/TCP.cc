#include "cache.h"
#include <iostream>
#include <vector>
#include <cstdint>


/*
This file implements the Tag Correlating Prefetcher described in the following paper:
https://mrmgroup.cs.princeton.edu/papers/tag_sequence.pdf
Using ChampSim; a cache simulator that is used to evaluate cache prefetchers.

This TCP operates by using a Tag History Table (THT) and a Pattern History Table (PHT) to predict the next tag to prefetch.
The TCP operates after the L1D cache and issues prefetches to L2.

Further notes on the general specifics of this particular implementation can be found in the comments below.

             John Iler 
  Student - Electrical Engineering
Texas A&M University College Station

Updated: 4/23/2024



Notes On Overall Functionality:

  Update function: 
                    locate tag_K (tag before missTag as defined by TCP documentation)
                    access THT w/tag_k (the previous missTag as defined by TCP documentation)
                    append miss_Tag to the accessed tag_Sequence row
                    when populating the THT (i.e. when the THT is not full):
                      append missTag to the first incomplete row
                    

                    access PHT with tag_k
                    set corresponding tag' = missTag
                    (tag' is the next Tag to prefetch as defined by TCP documentation)
                    when populating the PHT:
                      append tag_k and missTag to the first incomplete row with tag_k representing tag
                      and missTag representing tag'
                    
  Lookup Function:
                    locate PHT set containing missTag
                    fetch tag'
                    set predictedAddress = tag' + missIndex
                    issue predictedAdress to L2 cache using Prefetch_Line

Variables Necessary and/or mentioned in the TCP documentation:
  THT Variables:
                int sets_L1DCache
                int entriesPerRow_THT
                int size_tag
                int size_THT = sets_L1DCache * entriesPerRow_THT * size_tag
                string missIndex
                string missTag
                string missAddress? -> determine what is provided from champsim 

  PHT Variable:
                int sets_L1D
                int waysPerSet
                int size_PHT = sets_PHT * 2 * waysPerSet * size_tag

*/

// Constants for THT and PHT sizes

std::vector<uint64_t> misses; // stores all misses used in PHT update function

const int entriesPerRow_THT = 12; // Number of tags stored in each row of the THT. Can be adjusted.
const int sets_L1D = 64; // Number of sets in the L1D cache found in champsim_config.json
const int waysPerSet_PHT = 12; // Number of ways in each set of the L1D Cache. Found in champsim_config.json

const int tagSize = sets_L1D - 9; // Size of the tag in bits. 3 Bit offset and 6 bits for the index.

const int THT_SIZE = sets_L1D * entriesPerRow_THT * tagSize; // Reference Size of THT based on TCP documentation
const int PHT_SIZE = sets_L1D * waysPerSet_PHT * 2 * tagSize; // Reference Size of PHT based on TCP documentation

// Tag History Table (THT) Implementation
class TagHistoryTable {
private:
    // This structure simulates the rows in the tag history table
    struct THTEntry {
        uint64_t tags[entriesPerRow_THT]; 
    };

    // This establishes the THT as a vector of THTEntry structures
    // This is done to allow for easy indexing and updating of the THT "vertically"
    std::vector<THTEntry> THT_entries;


public:
    // Constructor to initialize the THT
    // Size is adjusted by the entries per row because THT_entries is a vector of THTEntry structures
    TagHistoryTable() : THT_entries(THT_SIZE / entriesPerRow_THT) {}


    // Function to update THT during cache miss
    void update(uint64_t missTag_fromCache) {
      uint64_t tag = misses[misses.size() - 1] / 1000000000; // Extracts the tag from the previous miss address
        
        // Iterate through the THT rows
        for(int i = 0; i < (THT_SIZE / entriesPerRow_THT); i++){
          
          // Iterate through the tags in each row
          for(int j = 0; j < entriesPerRow_THT; j++){
            
            // If the previous tag (tag_K) is already in the THT, shift the tags in that row and store the missTag at the end of the row
            if(THT_entries[i].tags[j] == tag){
              
              // Shift existing tags and store new tag at the end of the row
              for (int k = 0; k < entriesPerRow_THT-1; k++) {
                THT_entries[i].tags[k] = THT_entries[i].tags[k + 1];
              }

              // Store the missTag at the end of the row
              THT_entries[i].tags[entriesPerRow_THT - 1] = missTag_fromCache;
              return;
            }
          }
        }

        // if the previous process failed, then the tag_K is not in the THT
        // In this case, we need to find an incomplete row in the THT and store the missTag in that row
        
        // Iterate through the THT rows
        for(int i = 0; i < (THT_SIZE / entriesPerRow_THT); i++){
          
          // we want the first row that is incomplete (i.e. has an empty tag)
          // this is because we want to populate the history table chronologically
          if(THT_entries[i].tags[0] == 0){

            // add the missTag to the end of the incomplete row
            THT_entries[i].tags[entriesPerRow_THT - 1] = missTag_fromCache;
            return;
          }
        }
        
    }


    // Function to get tag sequence at a given index
    // Not used for the TCP, here as placeholder as it may be useful for debugging
    const uint64_t* get_tag_sequence(uint64_t index) const {
        return THT_entries[index].tags;
    }
};


// Pattern History Table (PHT) Implementation
class PatternHistoryTable {
private:

    // This structure simulates the rows in the pattern history table
    // Each row contains two tags tag and tag' as described in the TCP documentation
    struct PHTEntry {
        uint64_t tag_sequence[2];
        
    };
    // Once again, the PHT is implemented as a vector of PHTEntry structures
    // This allows for "vertical" access and updating of the PHT
    std::vector<PHTEntry> PHT_entries;


public:
    // Constructor to initialize the PHT
    // size is divided by 2 because each row contains two tags
    PatternHistoryTable() : PHT_entries(PHT_SIZE / 2) {}


    // Function to update PHT during cache miss
    void update(uint64_t missTag_fromCache) {
      int index = misses.size() - 2; // sets the index to the missAddress before the current miss (i.e. tag_k)
        
        // Iterate through the PHT entries
        for(int i = 0; i < (sets_L1D * waysPerSet_PHT); i++){
          
          // If the tag sequence at the first index of the PHT entry is equal to tag_k
          if(PHT_entries[i].tag_sequence[0] == misses[index] / 1000000000){ // this division provides the tag of the address
            
            // Set the second index of the PHT entry to the current missTag
            PHT_entries[i].tag_sequence[1] = missTag_fromCache;
            return;
          }
        }

        // If the previous process failed, then tag_k is not in the PHT
        // we need to find an empty row in the PHT to populate with the current missTag and tag_k
        for(int i = 0; i < (sets_L1D * waysPerSet_PHT); i++){
          
          // if an entry is 0 then the row is not full (i.e. incomplete)
          if(PHT_entries[i].tag_sequence[0] == 0){
            
            // populate the row with the current missTag and tag_k 
            // tag_k as tag and missTag as tag' (to use the language in the TCP documentation)
            PHT_entries[i].tag_sequence[0] = misses[index] / 1000000000;
            PHT_entries[i].tag_sequence[1] = missTag_fromCache;
            return;
          }
        }

    }

    // LookUP function from TCP documentation
    // search for the next tag after tag_k in the PHT
    uint64_t lookUp(uint64_t missTag_fromCache){
      for(int i = 0; i < sets_L1D; i++){

        // if the tag_k is in the PHT, return the next tag
        if(PHT_entries[i].tag_sequence[0] == missTag_fromCache){
          return PHT_entries[i].tag_sequence[1];
        }
      }

      // if the tag_k is not in the PHT, return 0
      return 0;
    }


};

// initialize the Pattern History Table and Tag History Table
PatternHistoryTable PHT_Main;
TagHistoryTable THT_Main;


void CACHE::prefetcher_initialize() {}

uint32_t CACHE::prefetcher_cache_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, bool useful_prefetch, uint8_t type, uint32_t metadata_in)
{
    // Extract the missTag, missIndex, and missOffset from the address
    // lowest 3 bits are the offset, next 6 bits are the index, and the rest are the tag
    uint64_t missTag = addr / 1000000000;
    uint64_t missIndex = (addr % 1000000000) / 1000;
    uint64_t missOffset = addr % 1000;

    // Store the miss in the misses vector
    misses.push_back(addr);

    // Update the THT and PHT
    THT_Main.update(missTag);
    PHT_Main.update(missTag);

    // Look up the PHT and get the next tag
    uint64_t pfTag = PHT_Main.lookUp(missTag);

    // Combine the next tag with the missIndex and missOffset to get the prefetch address
    uint64_t pfAddr = pfTag * 1000000000 + missIndex * 1000 + missOffset;

    if(pfTag == 0){
      prefetch_line(addr, false, metadata_in);
    }
    else{
      prefetch_line(pfAddr, true, metadata_in);
    }
    
  return metadata_in;
}

uint32_t CACHE::prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in)
{
  return metadata_in;
}

void CACHE::prefetcher_cycle_operate() {}

void CACHE::prefetcher_final_stats() {}