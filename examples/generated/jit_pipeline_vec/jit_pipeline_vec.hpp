// pyCircuit C++ emission (prototype)
#include <pyc/cpp/pyc_sim.hpp>

namespace pyc::gen {

struct JitPipelineVec {
  pyc::cpp::Wire<1> sys_clk{};
  pyc::cpp::Wire<1> sys_rst{};
  pyc::cpp::Wire<16> a{};
  pyc::cpp::Wire<16> b{};
  pyc::cpp::Wire<1> sel{};
  pyc::cpp::Wire<1> tag{};
  pyc::cpp::Wire<16> data{};
  pyc::cpp::Wire<8> lo8{};

  pyc::cpp::Wire<25> v1{};
  pyc::cpp::Wire<1> v2{};
  pyc::cpp::Wire<25> v3{};
  pyc::cpp::Wire<1> v4{};
  pyc::cpp::Wire<1> en__jit_pipeline_vec__L8{};
  pyc::cpp::Wire<16> a__jit_pipeline_vec__L10{};
  pyc::cpp::Wire<16> b__jit_pipeline_vec__L11{};
  pyc::cpp::Wire<1> sel__jit_pipeline_vec__L12{};
  pyc::cpp::Wire<16> v5{};
  pyc::cpp::Wire<16> sum___jit_pipeline_vec__L15{};
  pyc::cpp::Wire<16> v6{};
  pyc::cpp::Wire<16> x__jit_pipeline_vec__L16{};
  pyc::cpp::Wire<16> v7{};
  pyc::cpp::Wire<16> data__jit_pipeline_vec__L17{};
  pyc::cpp::Wire<1> v8{};
  pyc::cpp::Wire<1> tag__jit_pipeline_vec__L18{};
  pyc::cpp::Wire<8> v9{};
  pyc::cpp::Wire<8> lo8__jit_pipeline_vec__L19{};
  pyc::cpp::Wire<25> v10{};
  pyc::cpp::Wire<25> v11{};
  pyc::cpp::Wire<25> v12{};
  pyc::cpp::Wire<25> v13{};
  pyc::cpp::Wire<25> v14{};
  pyc::cpp::Wire<25> v15{};
  pyc::cpp::Wire<25> v16{};
  pyc::cpp::Wire<25> v17{};
  pyc::cpp::Wire<25> v18{};
  pyc::cpp::Wire<25> bus__jit_pipeline_vec__L22{};
  pyc::cpp::Wire<25> v19{};
  pyc::cpp::Wire<25> bus__jit_pipeline_vec__L26{};
  pyc::cpp::Wire<25> v20{};
  pyc::cpp::Wire<25> bus__jit_pipeline_vec__L26_2{};
  pyc::cpp::Wire<25> v21{};
  pyc::cpp::Wire<25> bus__jit_pipeline_vec__L26_3{};
  pyc::cpp::Wire<25> bus__jit_pipeline_vec__L25{};
  pyc::cpp::Wire<8> v22{};
  pyc::cpp::Wire<16> v23{};
  pyc::cpp::Wire<1> v24{};
  pyc::cpp::Wire<8> v25{};
  pyc::cpp::Wire<16> v26{};
  pyc::cpp::Wire<1> v27{};

  pyc::cpp::pyc_reg<25> v19_inst;
  pyc::cpp::pyc_reg<25> v20_inst;
  pyc::cpp::pyc_reg<25> v21_inst;

  JitPipelineVec() :
      v19_inst(sys_clk, sys_rst, en__jit_pipeline_vec__L8, bus__jit_pipeline_vec__L22, v3, v19),
      v20_inst(sys_clk, sys_rst, en__jit_pipeline_vec__L8, bus__jit_pipeline_vec__L26, v3, v20),
      v21_inst(sys_clk, sys_rst, en__jit_pipeline_vec__L8, bus__jit_pipeline_vec__L26_2, v3, v21) {
    eval();
  }

  inline void eval_comb_0() {
    v1 = pyc::cpp::Wire<25>(0ull);
    v2 = pyc::cpp::Wire<1>(1ull);
    v3 = v1;
    v4 = v2;
  }

  inline void eval_comb_1() {
    v10 = pyc::cpp::zext<25, 8>(lo8__jit_pipeline_vec__L19);
    v11 = (v3 | v10);
    v12 = pyc::cpp::zext<25, 16>(data__jit_pipeline_vec__L17);
    v13 = pyc::cpp::Wire<25>(v12.value() << 8ull);
    v14 = (v11 | v13);
    v15 = pyc::cpp::zext<25, 1>(tag__jit_pipeline_vec__L18);
    v16 = pyc::cpp::Wire<25>(v15.value() << 24ull);
    v17 = (v14 | v16);
    v18 = v17;
  }

  inline void eval_comb_2() {
    v22 = pyc::cpp::extract<8, 25>(bus__jit_pipeline_vec__L25, 0u);
    v23 = pyc::cpp::extract<16, 25>(bus__jit_pipeline_vec__L25, 8u);
    v24 = pyc::cpp::extract<1, 25>(bus__jit_pipeline_vec__L25, 24u);
    v25 = v22;
    v26 = v23;
    v27 = v24;
  }

  inline void eval_comb_pass() {
    eval_comb_0();
    en__jit_pipeline_vec__L8 = v4;
    a__jit_pipeline_vec__L10 = a;
    b__jit_pipeline_vec__L11 = b;
    sel__jit_pipeline_vec__L12 = sel;
    v5 = (a__jit_pipeline_vec__L10 + b__jit_pipeline_vec__L11);
    sum___jit_pipeline_vec__L15 = v5;
    v6 = (a__jit_pipeline_vec__L10 ^ b__jit_pipeline_vec__L11);
    x__jit_pipeline_vec__L16 = v6;
    v7 = (sel__jit_pipeline_vec__L12.toBool() ? sum___jit_pipeline_vec__L15 : x__jit_pipeline_vec__L16);
    data__jit_pipeline_vec__L17 = v7;
    v8 = pyc::cpp::Wire<1>((a__jit_pipeline_vec__L10 == b__jit_pipeline_vec__L11) ? 1u : 0u);
    tag__jit_pipeline_vec__L18 = v8;
    v9 = pyc::cpp::extract<8, 16>(data__jit_pipeline_vec__L17, 0u);
    lo8__jit_pipeline_vec__L19 = v9;
    eval_comb_1();
    bus__jit_pipeline_vec__L22 = v18;
    bus__jit_pipeline_vec__L26 = v19;
    bus__jit_pipeline_vec__L26_2 = v20;
    bus__jit_pipeline_vec__L26_3 = v21;
    bus__jit_pipeline_vec__L25 = bus__jit_pipeline_vec__L26_3;
    eval_comb_2();
  }

  void eval() {
    eval_comb_pass();
    tag = v27;
    data = v26;
    lo8 = v25;
  }

  void tick() {
    // Two-phase update: compute next state for all sequential elements,
    // then commit together. This avoids ordering artifacts between regs.
    // Phase 1: compute.
    v19_inst.tick_compute();
    v20_inst.tick_compute();
    v21_inst.tick_compute();
    // Phase 2: commit.
    v19_inst.tick_commit();
    v20_inst.tick_commit();
    v21_inst.tick_commit();
  }
};

} // namespace pyc::gen
