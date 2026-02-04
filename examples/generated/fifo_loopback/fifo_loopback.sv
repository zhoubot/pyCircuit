`include "pyc_add.sv"
`include "pyc_mux.sv"
`include "pyc_and.sv"
`include "pyc_or.sv"
`include "pyc_xor.sv"
`include "pyc_not.sv"
`include "pyc_reg.sv"
`include "pyc_fifo.sv"

`include "pyc_byte_mem.sv"

module FifoLoopback (
  input logic clk,
  input logic rst,
  input logic in_valid,
  input logic [7:0] in_data,
  input logic out_ready,
  output logic in_ready,
  output logic out_valid,
  output logic [7:0] out_data
);

logic in_valid__fifo_loopback__L10;
logic [7:0] in_data__fifo_loopback__L11;
logic out_ready__fifo_loopback__L12;
logic q__in_valid;
logic [7:0] q__in_data;
logic q__out_ready;
logic v1;
logic v2;
logic [7:0] v3;

assign in_valid__fifo_loopback__L10 = in_valid;
assign in_data__fifo_loopback__L11 = in_data;
assign out_ready__fifo_loopback__L12 = out_ready;
pyc_fifo #(.WIDTH(8), .DEPTH(2)) v1_inst (
  .clk(clk),
  .rst(rst),
  .in_valid(q__in_valid),
  .in_ready(v1),
  .in_data(q__in_data),
  .out_valid(v2),
  .out_ready(q__out_ready),
  .out_data(v3)
);
assign q__in_valid = in_valid__fifo_loopback__L10;
assign q__in_data = in_data__fifo_loopback__L11;
assign q__out_ready = out_ready__fifo_loopback__L12;
assign in_ready = v1;
assign out_valid = v2;
assign out_data = v3;

endmodule

