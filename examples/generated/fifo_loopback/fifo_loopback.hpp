// pyCircuit C++ emission (prototype)
#include <pyc/cpp/pyc_sim.hpp>

namespace pyc::gen {

struct FifoLoopback {
  pyc::cpp::Wire<1> clk{};
  pyc::cpp::Wire<1> rst{};
  pyc::cpp::Wire<1> in_valid{};
  pyc::cpp::Wire<8> in_data{};
  pyc::cpp::Wire<1> out_ready{};
  pyc::cpp::Wire<1> in_ready{};
  pyc::cpp::Wire<1> out_valid{};
  pyc::cpp::Wire<8> out_data{};

  pyc::cpp::Wire<1> in_valid__fifo_loopback__L10{};
  pyc::cpp::Wire<8> in_data__fifo_loopback__L11{};
  pyc::cpp::Wire<1> out_ready__fifo_loopback__L12{};
  pyc::cpp::Wire<1> q__in_valid{};
  pyc::cpp::Wire<8> q__in_data{};
  pyc::cpp::Wire<1> q__out_ready{};
  pyc::cpp::Wire<1> v1{};
  pyc::cpp::Wire<1> v2{};
  pyc::cpp::Wire<8> v3{};

  pyc::cpp::pyc_fifo<8, 2> v1_inst;

  FifoLoopback() :
      v1_inst(clk, rst, q__in_valid, v1, q__in_data, v2, q__out_ready, v3) {
    eval();
  }

  inline void eval_comb_pass() {
    in_valid__fifo_loopback__L10 = in_valid;
    in_data__fifo_loopback__L11 = in_data;
    out_ready__fifo_loopback__L12 = out_ready;
    q__in_valid = in_valid__fifo_loopback__L10;
    q__in_data = in_data__fifo_loopback__L11;
    q__out_ready = out_ready__fifo_loopback__L12;
  }

  void eval() {
    eval_comb_pass();
    for (unsigned _i = 0; _i < 1u; ++_i) {
      v1_inst.eval();
      eval_comb_pass();
    }
    in_ready = v1;
    out_valid = v2;
    out_data = v3;
  }

  void tick() {
    // Two-phase update: compute next state for all sequential elements,
    // then commit together. This avoids ordering artifacts between regs.
    // Phase 1: compute.
    v1_inst.tick_compute();
    // Phase 2: commit.
    v1_inst.tick_commit();
  }
};

} // namespace pyc::gen
