// pyCircuit C++ emission (prototype)
#include <pyc/cpp/pyc_sim.hpp>

namespace pyc::gen {

struct Counter {
  pyc::cpp::Wire<1> clk{};
  pyc::cpp::Wire<1> rst{};
  pyc::cpp::Wire<1> en{};
  pyc::cpp::Wire<8> count{};

  pyc::cpp::Wire<8> v1{};
  pyc::cpp::Wire<8> v2{};
  pyc::cpp::Wire<8> v3{};
  pyc::cpp::Wire<8> v4{};
  pyc::cpp::Wire<1> en__counter__L9{};
  pyc::cpp::Wire<8> count__next{};
  pyc::cpp::Wire<8> v5{};
  pyc::cpp::Wire<8> count{};
  pyc::cpp::Wire<8> count__counter__L11{};
  pyc::cpp::Wire<8> v6{};

  pyc::cpp::pyc_reg<8> v5_inst;

  Counter() :
      v5_inst(clk, rst, en__counter__L9, count__next, v4, v5) {
    eval();
  }

  inline void eval_comb_0() {
    v1 = pyc::cpp::Wire<8>(1ull);
    v2 = pyc::cpp::Wire<8>(0ull);
    v3 = v1;
    v4 = v2;
  }

  inline void eval_comb_pass() {
    eval_comb_0();
    en__counter__L9 = en;
    count = v5;
    count__counter__L11 = count;
    v6 = (count__counter__L11 + v3);
    count__next = v6;
  }

  void eval() {
    eval_comb_pass();
    count = count__counter__L11;
  }

  void tick() {
    // Two-phase update: compute next state for all sequential elements,
    // then commit together. This avoids ordering artifacts between regs.
    // Phase 1: compute.
    v5_inst.tick_compute();
    // Phase 2: commit.
    v5_inst.tick_commit();
  }
};

} // namespace pyc::gen
