`include "pyc_add.sv"
`include "pyc_mux.sv"
`include "pyc_and.sv"
`include "pyc_or.sv"
`include "pyc_xor.sv"
`include "pyc_not.sv"
`include "pyc_reg.sv"
`include "pyc_fifo.sv"

`include "pyc_byte_mem.sv"

module WireOps (
  input logic sys_clk,
  input logic sys_rst,
  input logic [7:0] a,
  input logic [7:0] b,
  input logic sel,
  output logic [7:0] y
);

logic [7:0] v1;
logic v2;
logic [7:0] v3;
logic v4;
logic [7:0] a__wire_ops__L9;
logic [7:0] b__wire_ops__L10;
logic sel__wire_ops__L11;
logic [7:0] v5;
logic [7:0] v6;
logic [7:0] v7;
logic [7:0] v8;
logic [7:0] y__wire_ops__L13;
logic en__wire_ops__L15;
logic [7:0] v9;
logic [7:0] r__wire_ops__L16;

assign v1 = 8'd0;
assign v2 = 1'd1;
assign v3 = v1;
assign v4 = v2;
assign a__wire_ops__L9 = a;
assign b__wire_ops__L10 = b;
assign sel__wire_ops__L11 = sel;
assign v5 = (a__wire_ops__L9 & b__wire_ops__L10);
assign v6 = (a__wire_ops__L9 ^ b__wire_ops__L10);
assign v7 = (sel__wire_ops__L11 ? v5 : v6);
assign v8 = v7;
assign y__wire_ops__L13 = v8;
assign en__wire_ops__L15 = v4;
pyc_reg #(.WIDTH(8)) v9_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(en__wire_ops__L15),
  .d(y__wire_ops__L13),
  .init(v3),
  .q(v9)
);
assign r__wire_ops__L16 = v9;
assign y = r__wire_ops__L16;

endmodule

