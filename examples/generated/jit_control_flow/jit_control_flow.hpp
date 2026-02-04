// pyCircuit C++ emission (prototype)
#include <pyc/cpp/pyc_sim.hpp>

namespace pyc::gen {

struct JitControlFlow {
  pyc::cpp::Wire<8> a{};
  pyc::cpp::Wire<8> b{};
  pyc::cpp::Wire<8> out{};

  pyc::cpp::Wire<8> v1{};
  pyc::cpp::Wire<8> v2{};
  pyc::cpp::Wire<8> v3{};
  pyc::cpp::Wire<8> v4{};
  pyc::cpp::Wire<8> a__jit_control_flow__L7{};
  pyc::cpp::Wire<8> b__jit_control_flow__L8{};
  pyc::cpp::Wire<8> v5{};
  pyc::cpp::Wire<8> x__jit_control_flow__L10{};
  pyc::cpp::Wire<1> v6{};
  pyc::cpp::Wire<8> v7{};
  pyc::cpp::Wire<1> v8{};
  pyc::cpp::Wire<8> v9{};
  pyc::cpp::Wire<8> x__jit_control_flow__L12{};
  pyc::cpp::Wire<8> v10{};
  pyc::cpp::Wire<8> x__jit_control_flow__L14{};
  pyc::cpp::Wire<8> v11{};
  pyc::cpp::Wire<8> x__jit_control_flow__L11{};
  pyc::cpp::Wire<8> acc__jit_control_flow__L16{};
  pyc::cpp::Wire<8> v12{};
  pyc::cpp::Wire<8> acc__jit_control_flow__L18{};
  pyc::cpp::Wire<8> v13{};
  pyc::cpp::Wire<8> acc__jit_control_flow__L18_2{};
  pyc::cpp::Wire<8> v14{};
  pyc::cpp::Wire<8> acc__jit_control_flow__L18_3{};
  pyc::cpp::Wire<8> v15{};
  pyc::cpp::Wire<8> acc__jit_control_flow__L18_4{};
  pyc::cpp::Wire<8> acc__jit_control_flow__L17{};


  JitControlFlow() {
    eval();
  }

  inline void eval_comb_0() {
    v1 = pyc::cpp::Wire<8>(2ull);
    v2 = pyc::cpp::Wire<8>(1ull);
    v3 = v1;
    v4 = v2;
  }

  inline void eval_comb_1() {
    v6 = pyc::cpp::Wire<1>((a__jit_control_flow__L7 == b__jit_control_flow__L8) ? 1u : 0u);
    v7 = (x__jit_control_flow__L10 + v4);
    v8 = v6;
    v9 = v7;
  }

  inline void eval_comb_pass() {
    eval_comb_0();
    a__jit_control_flow__L7 = a;
    b__jit_control_flow__L8 = b;
    v5 = (a__jit_control_flow__L7 + b__jit_control_flow__L8);
    x__jit_control_flow__L10 = v5;
    eval_comb_1();
    x__jit_control_flow__L12 = v9;
    v10 = (x__jit_control_flow__L10 + v3);
    x__jit_control_flow__L14 = v10;
    v11 = (v8.toBool() ? x__jit_control_flow__L12 : x__jit_control_flow__L14);
    x__jit_control_flow__L11 = v11;
    acc__jit_control_flow__L16 = x__jit_control_flow__L11;
    v12 = (acc__jit_control_flow__L16 + v4);
    acc__jit_control_flow__L18 = v12;
    v13 = (acc__jit_control_flow__L18 + v4);
    acc__jit_control_flow__L18_2 = v13;
    v14 = (acc__jit_control_flow__L18_2 + v4);
    acc__jit_control_flow__L18_3 = v14;
    v15 = (acc__jit_control_flow__L18_3 + v4);
    acc__jit_control_flow__L18_4 = v15;
    acc__jit_control_flow__L17 = acc__jit_control_flow__L18_4;
  }

  void eval() {
    eval_comb_pass();
    out = acc__jit_control_flow__L17;
  }

  void tick() {
    // Two-phase update: compute next state for all sequential elements,
    // then commit together. This avoids ordering artifacts between regs.
    // Phase 1: compute.
    // Phase 2: commit.
  }
};

} // namespace pyc::gen
