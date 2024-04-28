#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <queue>
#include "cache.h"
#include "msl/lru_table.h"

#include <fstream>
#include <iostream>

/*main points:
when miss happens, dont immediately prefetchm using pred addr. delay prefetch until predicted time when the prefetch should be issued.
learn trigger PCs for timing prefetches for the target PCs
issue prefetch, store in IPT. When prefetch line inserted in cache, the trigger pc is used from IPT and recorded in a RRPCQ
when access causes a miss, PCs in RRPCSQ are linked to the target PCs in the target table. 
^^^ helps learn relationship between trigger and target.*/

namespace {
    std::ofstream debug_file("t_skid_debug.txt");
    uint64_t debug_counter = 0; //counter for debugging
    const uint64_t interval = 100; //print ever x iterations.

    void debug_print(const std::string& msg){
        if(debug_counter % interval == 0){
            debug_file << msg << std::endl;
        }
        debug_counter++; //up the increment always.
    }

struct tracker {
    struct tracker_entry {
        //stores IP, last cl address, and stride between last two cl addresses
        uint64_t ip = 0;
        uint64_t last_cl_addr = 0;
        int64_t last_stride = 0;

        auto index() const { return ip; }
        auto tag() const { return ip; }
    };

    struct lookahead_entry {
        //stores currently ttargeted pf address, stride used for prefetching
        //and the remaining degree of prefetching
        uint64_t address = 0;
        int64_t stride = 0;
        int degree = 0;
    };

    struct target_table_entry {
        //for prefetching decisions.
        //stores target and trigger pc
        uint64_t target_pc;
        uint64_t trigger_pc;
    };

    struct addr_pred_table_entry {
        //similar to lookahead.
        //TODO: remove redundant
        uint64_t last_addr;
        int64_t stride;
        int degree;
    };

    struct inflight_prefetch_entry {
        //prefetcches currently used. 
        //stores IP that triggered the prefetch and the prefetch address
        uint64_t trigger_pc;
        uint64_t prefetch_addr;
    };

    constexpr static std::size_t TRACKER_SETS = 256;
    constexpr static std::size_t TRACKER_WAYS = 4;
    constexpr static int PREFETCH_DEGREE = 3;
    constexpr static std::size_t TARGET_TABLE_SIZE = 256;
    constexpr static std::size_t ADDR_PRED_TABLE_SIZE = 256;
    constexpr static std::size_t IPT_SIZE = 16;
    constexpr static std::size_t RRPCQ_SIZE = 16;

    std::optional<lookahead_entry> active_lookahead;
    champsim::msl::lru_table<tracker_entry> table{TRACKER_SETS, TRACKER_WAYS}; //"last recently used"

    std::map<uint64_t, target_table_entry> target_table; //maps trigger to target PCs
    std::map<uint64_t, addr_pred_table_entry> addr_pred_table; //store last address, stride, and degree for each target pc for address prediction
    std::vector<inflight_prefetch_entry> inflight_prefetch_table; //records issues prefetches that have not yet been filled
    std::queue<uint64_t> recent_request_pc_queue; //store recenetly seen trigger pcs

    //prefetch based on ip and cl address
    /*Based on a given IP and cl address
    Decide whetehr to initiate a prefetche
    Check for a hit in the LRU table
    calculate stride
    checks for repeated patters
    IF CONDITIONS MET: trigger prefetch and update prediction table*/
    void initiate_lookahead(uint64_t ip, uint64_t cl_addr, CACHE* cache) {
        int64_t stride = 0;
        auto found = table.check_hit({ip, cl_addr, stride}); 
        debug_print("(1)initiate_lookahead: you are in the function");
        if (found.has_value()) {
            debug_print("(2)initiate_lookahead: you are in first if");
            debug_print("calculating stride... with 1: " + std::to_string(cl_addr) + " and 2: " + std::to_string(found->last_cl_addr));
            stride = static_cast<int64_t>(cl_addr) - static_cast<int64_t>(found->last_cl_addr);

            if (stride != 0 && stride == found->last_stride) {
                debug_print("(3)initiate_lookahead: you are in second if");
                auto it = target_table.find(ip);
                if (it != target_table.end()) {
                    debug_print("(4)initiate_lookahead: you are in third if");
                    uint64_t target_pc = it->second.target_pc;
                    auto pred_it = addr_pred_table.find(target_pc);
                    if (pred_it != addr_pred_table.end()) {
                        debug_print("(5)initiate_lookahead: you are in fourth if PREFETCH ISSUING *************");
                        uint64_t pf_addr = pred_it->second.last_addr + pred_it->second.stride;
                        int degree = pred_it->second.degree; //degree gets adjusted in advance_lookahead
                        issue_prefetch(cache, ip, pf_addr, degree);
                    }
                }
            }

            // Update address prediction table
            //mapping IP to its memory access data
            auto pred_it = addr_pred_table.find(ip);
            if (pred_it != addr_pred_table.end()) {
                //entry exists, update
                debug_print("(6)initiate_lookahead: updating addr_pred_table entry");
                pred_it->second.last_addr = cl_addr;
                pred_it->second.stride = stride;
            } else {
                debug_print("(7)initiate_lookahead: creating new addr_pred_table entry");
                addr_pred_table[ip] = {cl_addr, stride, PREFETCH_DEGREE};
            }
        }

        table.fill({ip, cl_addr, stride}); //update the lru table
    }

    /*called by initiate_lookahead to send a prefetch request in MSHR (miss status holding register)
    IF OCCUPANCY is low, which means there is bandwidth*/
    //logic similar to stride.
    void issue_prefetch(CACHE* cache, uint64_t trigger_pc, uint64_t pf_addr, int degree) {
        debug_print("(8)issue_prefetch: you are in the function");
        bool success = cache->prefetch_line(pf_addr, (cache->get_mshr_occupancy_ratio() < 0.5), 0);
        if (success) {
            debug_print("(9)issue_prefetch: prefetch issued ADD TO IPT");
            inflight_prefetch_table.push_back({trigger_pc, pf_addr});
        }
    }

    /*Called by every cycle to update and manage prefetching
    based on the status of prefetches and memory access events*/
    void advance_lookahead(CACHE* cache) {
        //Perform timing learning
        //iterate through MSHR, MSHR tracks prefetches not yet completed in IPT
        debug_print("(10)advance_lookahead: you are in the function");
        for (auto it = cache->MSHR.begin(); it != cache->MSHR.end(); ++it) {
            //LEARNING TIMING AT PREFETCH FILL
            if (it->event_cycle <= cache->current_cycle) {
                debug_print("(11)advance_lookahead: prefetch completed");
                //if current cycle is equal or passed event cycle... this means prefetch is completed
                //if prefetch is completed, remove from inflight prefetch table
                //event cycle is the cycle pf supposed to complete
                //current cycle is the current cycle of the simulation.
                auto fill_address = it->address;
                auto fill_it = std::find_if(inflight_prefetch_table.begin(), inflight_prefetch_table.end(), [fill_address](const auto& entry) { return entry.prefetch_addr == fill_address; }); //find the prefetch address in the inflight prefetch table
                if (fill_it != inflight_prefetch_table.end()) {
                    debug_print("(12)advance_lookahead: prefetch found in IPT");
                    //if found in IPT, push trigger pc to RRPCQ
                    uint64_t trigger_pc = fill_it->trigger_pc;
                    recent_request_pc_queue.push(trigger_pc); //PUSH HERE
                    if (recent_request_pc_queue.size() > RRPCQ_SIZE) {
                        debug_print("(13)advance_lookahead: RRPCQ size overflow");
                        recent_request_pc_queue.pop(); //if size overflow, POP oldest entry.
                    }
                    inflight_prefetch_table.erase(fill_it); 
                }
            }
        }

        //Perform target PC linking
        //LEARNING TIMING AT CACHE ACCESS
        auto container = cache->get_inflight_tag_check(); //cahce tag check q holds info ongoing mem access that are not yet completed
        
        for (auto it = container.begin(); it != container.end(); ++it) {
            //iterate through unresolved mem accesses
            const auto& tag_entry = *it;
            if (tag_entry.event_cycle <= cache->current_cycle && !tag_entry.is_translated && tag_entry.type == access_type::LOAD) {
                //pf occured or occuring
                //has not translated to physical mem address
                //it's a load
                debug_print("(14)advance_lookahead: target pc linking FIRST IF");
                uint64_t target_pc = tag_entry.ip;
                while (!recent_request_pc_queue.empty()) {
                    //iterate through RRPCQ till empty
                    uint64_t trigger_pc = recent_request_pc_queue.front(); //get the trigger pc
                    recent_request_pc_queue.pop(); //remove it
                    target_table[trigger_pc] = {target_pc, trigger_pc}; //link trigger with target
                    debug_print("(15)advance_lookahead: target pc linked");
                }
            }
        }

        //Update degree in address prediction table. dynamically adjust prefetch degree
        for (auto it = inflight_prefetch_table.begin(); it != inflight_prefetch_table.end(); ++it) {
            //iterate over IPT
            auto& entry = *it; //in-progress prefetch in IPT

            auto pred_it = addr_pred_table.find(entry.trigger_pc); //look for entry's trigger pc in addr_pred_table
            if (pred_it != addr_pred_table.end()) {
                //if entry in addr_pred_table... which means prefetch was issued.
                debug_print("(16)advance_lookahead: updating degree in addr_pred_table FIRST IF");
                //decrement degree, make sure stay above 1.
                if (pred_it->second.degree > 1) {
                    pred_it->second.degree -= 1;
                    debug_print("(17)advance_lookahead: degree updated (-1)");
                } 
                else {
                    // Otherwise, set the degree to 1 to ensure it does not fall below this value
                    pred_it->second.degree = 1;
                    debug_print("(18)advance_lookahead: degree updated (=1)");
                }
            }
        }
    }
};

std::map<CACHE*, tracker> trackers;

} // namespace

void CACHE::prefetcher_initialize() {}

void CACHE::prefetcher_cycle_operate() {
    ::trackers[this].advance_lookahead(this);
}

uint32_t CACHE::prefetcher_cache_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, bool useful_prefetch, uint8_t type, uint32_t metadata_in) {
    ::trackers[this].initiate_lookahead(ip, addr >> LOG2_BLOCK_SIZE, this);
    return metadata_in;
}

uint32_t CACHE::prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in) {
    return metadata_in;
}

void CACHE::prefetcher_final_stats() {
    //contents of inflight_prefetch_table
    std::cout << "Inflight Prefetch Table Contents:" << std::endl;
    for (size_t i = 0; i < ::trackers[this].inflight_prefetch_table.size(); ++i) {
        const auto& entry = ::trackers[this].inflight_prefetch_table[i];
        std::cout << "Trigger PC: " << entry.trigger_pc << ", Prefetch Address: " << entry.prefetch_addr << std::endl;
    }

    //addr_pred_table
    std::cout << "Address Prediction Table Contents:" << std::endl;
    for (auto it = ::trackers[this].addr_pred_table.begin(); it != ::trackers[this].addr_pred_table.end(); ++it) {
        const auto& entry = *it;
        std::cout << "IP: " << entry.first << ", Last Address: " << entry.second.last_addr 
                  << ", Stride: " << entry.second.stride << ", Degree: " << entry.second.degree << std::endl;
    }

    //target_table
    std::cout << "Target Table Contents:" << std::endl;
    for (auto it = ::trackers[this].target_table.begin(); it != ::trackers[this].target_table.end(); ++it) {
        const auto& entry = *it;
        std::cout << "Trigger PC: " << entry.first << ", Target PC: " << entry.second.target_pc << std::endl;
    }
}
