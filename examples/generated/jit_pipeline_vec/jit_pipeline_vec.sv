`include "pyc_add.sv"
`include "pyc_mux.sv"
`include "pyc_and.sv"
`include "pyc_or.sv"
`include "pyc_xor.sv"
`include "pyc_not.sv"
`include "pyc_reg.sv"
`include "pyc_fifo.sv"

`include "pyc_byte_mem.sv"

module JitPipelineVec (
  input logic sys_clk,
  input logic sys_rst,
  input logic [15:0] a,
  input logic [15:0] b,
  input logic sel,
  output logic tag,
  output logic [15:0] data,
  output logic [7:0] lo8
);

logic [24:0] v1;
logic v2;
logic [24:0] v3;
logic v4;
logic en__jit_pipeline_vec__L8;
logic [15:0] a__jit_pipeline_vec__L10;
logic [15:0] b__jit_pipeline_vec__L11;
logic sel__jit_pipeline_vec__L12;
logic [15:0] v5;
logic [15:0] sum___jit_pipeline_vec__L15;
logic [15:0] v6;
logic [15:0] x__jit_pipeline_vec__L16;
logic [15:0] v7;
logic [15:0] data__jit_pipeline_vec__L17;
logic v8;
logic tag__jit_pipeline_vec__L18;
logic [7:0] v9;
logic [7:0] lo8__jit_pipeline_vec__L19;
logic [24:0] v10;
logic [24:0] v11;
logic [24:0] v12;
logic [24:0] v13;
logic [24:0] v14;
logic [24:0] v15;
logic [24:0] v16;
logic [24:0] v17;
logic [24:0] v18;
logic [24:0] bus__jit_pipeline_vec__L22;
logic [24:0] v19;
logic [24:0] bus__jit_pipeline_vec__L26;
logic [24:0] v20;
logic [24:0] bus__jit_pipeline_vec__L26_2;
logic [24:0] v21;
logic [24:0] bus__jit_pipeline_vec__L26_3;
logic [24:0] bus__jit_pipeline_vec__L25;
logic [7:0] v22;
logic [15:0] v23;
logic v24;
logic [7:0] v25;
logic [15:0] v26;
logic v27;

assign v1 = 25'd0;
assign v2 = 1'd1;
assign v3 = v1;
assign v4 = v2;
assign en__jit_pipeline_vec__L8 = v4;
assign a__jit_pipeline_vec__L10 = a;
assign b__jit_pipeline_vec__L11 = b;
assign sel__jit_pipeline_vec__L12 = sel;
pyc_add #(.WIDTH(16)) v5_inst (
  .a(a__jit_pipeline_vec__L10),
  .b(b__jit_pipeline_vec__L11),
  .y(v5)
);
assign sum___jit_pipeline_vec__L15 = v5;
pyc_xor #(.WIDTH(16)) v6_inst (
  .a(a__jit_pipeline_vec__L10),
  .b(b__jit_pipeline_vec__L11),
  .y(v6)
);
assign x__jit_pipeline_vec__L16 = v6;
pyc_mux #(.WIDTH(16)) v7_inst (
  .sel(sel__jit_pipeline_vec__L12),
  .a(sum___jit_pipeline_vec__L15),
  .b(x__jit_pipeline_vec__L16),
  .y(v7)
);
assign data__jit_pipeline_vec__L17 = v7;
assign v8 = (a__jit_pipeline_vec__L10 == b__jit_pipeline_vec__L11);
assign tag__jit_pipeline_vec__L18 = v8;
assign v9 = data__jit_pipeline_vec__L17[7:0];
assign lo8__jit_pipeline_vec__L19 = v9;
assign v10 = {{17{1'b0}}, lo8__jit_pipeline_vec__L19};
assign v11 = (v3 | v10);
assign v12 = {{9{1'b0}}, data__jit_pipeline_vec__L17};
assign v13 = (v12 << 8);
assign v14 = (v11 | v13);
assign v15 = {{24{1'b0}}, tag__jit_pipeline_vec__L18};
assign v16 = (v15 << 24);
assign v17 = (v14 | v16);
assign v18 = v17;
assign bus__jit_pipeline_vec__L22 = v18;
pyc_reg #(.WIDTH(25)) v19_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(en__jit_pipeline_vec__L8),
  .d(bus__jit_pipeline_vec__L22),
  .init(v3),
  .q(v19)
);
assign bus__jit_pipeline_vec__L26 = v19;
pyc_reg #(.WIDTH(25)) v20_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(en__jit_pipeline_vec__L8),
  .d(bus__jit_pipeline_vec__L26),
  .init(v3),
  .q(v20)
);
assign bus__jit_pipeline_vec__L26_2 = v20;
pyc_reg #(.WIDTH(25)) v21_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(en__jit_pipeline_vec__L8),
  .d(bus__jit_pipeline_vec__L26_2),
  .init(v3),
  .q(v21)
);
assign bus__jit_pipeline_vec__L26_3 = v21;
assign bus__jit_pipeline_vec__L25 = bus__jit_pipeline_vec__L26_3;
assign v22 = bus__jit_pipeline_vec__L25[7:0];
assign v23 = bus__jit_pipeline_vec__L25[23:8];
assign v24 = bus__jit_pipeline_vec__L25[24];
assign v25 = v22;
assign v26 = v23;
assign v27 = v24;
assign tag = v27;
assign data = v26;
assign lo8 = v25;

endmodule

