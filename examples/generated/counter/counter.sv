`include "pyc_add.sv"
`include "pyc_mux.sv"
`include "pyc_and.sv"
`include "pyc_or.sv"
`include "pyc_xor.sv"
`include "pyc_not.sv"
`include "pyc_reg.sv"
`include "pyc_fifo.sv"

`include "pyc_byte_mem.sv"

module Counter (
  input logic clk,
  input logic rst,
  input logic en,
  output logic [7:0] count
);

logic [7:0] v1;
logic [7:0] v2;
logic [7:0] v3;
logic [7:0] v4;
logic en__counter__L9;
logic [7:0] count__next;
logic [7:0] v5;
logic [7:0] count;
logic [7:0] count__counter__L11;
logic [7:0] v6;

assign v1 = 8'd1;
assign v2 = 8'd0;
assign v3 = v1;
assign v4 = v2;
assign en__counter__L9 = en;
pyc_reg #(.WIDTH(8)) v5_inst (
  .clk(clk),
  .rst(rst),
  .en(en__counter__L9),
  .d(count__next),
  .init(v4),
  .q(v5)
);
assign count = v5;
assign count__counter__L11 = count;
pyc_add #(.WIDTH(8)) v6_inst (
  .a(count__counter__L11),
  .b(v3),
  .y(v6)
);
assign count__next = v6;
assign count = count__counter__L11;

endmodule

