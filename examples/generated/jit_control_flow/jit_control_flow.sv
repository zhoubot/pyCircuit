`include "pyc_add.sv"
`include "pyc_mux.sv"
`include "pyc_and.sv"
`include "pyc_or.sv"
`include "pyc_xor.sv"
`include "pyc_not.sv"
`include "pyc_reg.sv"
`include "pyc_fifo.sv"

`include "pyc_byte_mem.sv"

module JitControlFlow (
  input logic [7:0] a,
  input logic [7:0] b,
  output logic [7:0] out
);

logic [7:0] v1;
logic [7:0] v2;
logic [7:0] v3;
logic [7:0] v4;
logic [7:0] a__jit_control_flow__L7;
logic [7:0] b__jit_control_flow__L8;
logic [7:0] v5;
logic [7:0] x__jit_control_flow__L10;
logic v6;
logic [7:0] v7;
logic v8;
logic [7:0] v9;
logic [7:0] x__jit_control_flow__L12;
logic [7:0] v10;
logic [7:0] x__jit_control_flow__L14;
logic [7:0] v11;
logic [7:0] x__jit_control_flow__L11;
logic [7:0] acc__jit_control_flow__L16;
logic [7:0] v12;
logic [7:0] acc__jit_control_flow__L18;
logic [7:0] v13;
logic [7:0] acc__jit_control_flow__L18_2;
logic [7:0] v14;
logic [7:0] acc__jit_control_flow__L18_3;
logic [7:0] v15;
logic [7:0] acc__jit_control_flow__L18_4;
logic [7:0] acc__jit_control_flow__L17;

assign v1 = 8'd2;
assign v2 = 8'd1;
assign v3 = v1;
assign v4 = v2;
assign a__jit_control_flow__L7 = a;
assign b__jit_control_flow__L8 = b;
pyc_add #(.WIDTH(8)) v5_inst (
  .a(a__jit_control_flow__L7),
  .b(b__jit_control_flow__L8),
  .y(v5)
);
assign x__jit_control_flow__L10 = v5;
assign v6 = (a__jit_control_flow__L7 == b__jit_control_flow__L8);
assign v7 = (x__jit_control_flow__L10 + v4);
assign v8 = v6;
assign v9 = v7;
assign x__jit_control_flow__L12 = v9;
pyc_add #(.WIDTH(8)) v10_inst (
  .a(x__jit_control_flow__L10),
  .b(v3),
  .y(v10)
);
assign x__jit_control_flow__L14 = v10;
pyc_mux #(.WIDTH(8)) v11_inst (
  .sel(v8),
  .a(x__jit_control_flow__L12),
  .b(x__jit_control_flow__L14),
  .y(v11)
);
assign x__jit_control_flow__L11 = v11;
assign acc__jit_control_flow__L16 = x__jit_control_flow__L11;
pyc_add #(.WIDTH(8)) v12_inst (
  .a(acc__jit_control_flow__L16),
  .b(v4),
  .y(v12)
);
assign acc__jit_control_flow__L18 = v12;
pyc_add #(.WIDTH(8)) v13_inst (
  .a(acc__jit_control_flow__L18),
  .b(v4),
  .y(v13)
);
assign acc__jit_control_flow__L18_2 = v13;
pyc_add #(.WIDTH(8)) v14_inst (
  .a(acc__jit_control_flow__L18_2),
  .b(v4),
  .y(v14)
);
assign acc__jit_control_flow__L18_3 = v14;
pyc_add #(.WIDTH(8)) v15_inst (
  .a(acc__jit_control_flow__L18_3),
  .b(v4),
  .y(v15)
);
assign acc__jit_control_flow__L18_4 = v15;
assign acc__jit_control_flow__L17 = acc__jit_control_flow__L18_4;
assign out = acc__jit_control_flow__L17;

endmodule

