module tb_janus_bcc_ooo_pyc;
  logic clk;
  logic rst;

  logic [63:0] boot_pc;
  logic [63:0] boot_sp;

  logic halted;
  logic [63:0] cycles;
  logic [63:0] pc;
  logic [63:0] fpc;
  logic [63:0] a0;
  logic [63:0] a1;
  logic [63:0] ra;
  logic [63:0] sp;

  logic [5:0] commit_op;
  logic commit_fire;
  logic [63:0] commit_value;
  logic [1:0] commit_dst_kind;
  logic [5:0] commit_dst_areg;
  logic [5:0] commit_pdst;
  logic [4:0] rob_count;

  logic [63:0] ct0;
  logic [63:0] cu0;
  logic [63:0] st0;
  logic [63:0] su0;

  logic issue_fire;
  logic [5:0] issue_op;
  logic [63:0] issue_pc;
  logic [3:0] issue_rob;
  logic [5:0] issue_sl;
  logic [5:0] issue_sr;
  logic [5:0] issue_sp;
  logic [5:0] issue_pdst;
  logic [63:0] issue_sl_val;
  logic [63:0] issue_sr_val;
  logic [63:0] issue_sp_val;
  logic issue_is_load;
  logic issue_is_store;
  logic store_pending;
  logic store_pending_older;
  logic [63:0] mem_raddr;
  logic dispatch_fire;
  logic [5:0] dec_op;

  janus_bcc_ooo_pyc dut (
      .clk(clk),
      .rst(rst),
      .boot_pc(boot_pc),
      .boot_sp(boot_sp),
      .halted(halted),
      .cycles(cycles),
      .pc(pc),
      .fpc(fpc),
      .a0(a0),
      .a1(a1),
      .ra(ra),
      .sp(sp),
      .commit_op(commit_op),
      .commit_fire(commit_fire),
      .commit_value(commit_value),
      .commit_dst_kind(commit_dst_kind),
      .commit_dst_areg(commit_dst_areg),
      .commit_pdst(commit_pdst),
      .rob_count(rob_count),
      .ct0(ct0),
      .cu0(cu0),
      .st0(st0),
      .su0(su0),
      .issue_fire(issue_fire),
      .issue_op(issue_op),
      .issue_pc(issue_pc),
      .issue_rob(issue_rob),
      .issue_sl(issue_sl),
      .issue_sr(issue_sr),
      .issue_sp(issue_sp),
      .issue_pdst(issue_pdst),
      .issue_sl_val(issue_sl_val),
      .issue_sr_val(issue_sr_val),
      .issue_sp_val(issue_sp_val),
      .issue_is_load(issue_is_load),
      .issue_is_store(issue_is_store),
      .store_pending(store_pending),
      .store_pending_older(store_pending_older),
      .mem_raddr(mem_raddr),
      .dispatch_fire(dispatch_fire),
      .dec_op(dec_op)
  );

  always #5 clk = ~clk;

  function automatic logic [31:0] mem_read32(input int unsigned addr);
    mem_read32 = {dut.mem.mem[addr + 3], dut.mem.mem[addr + 2], dut.mem.mem[addr + 1], dut.mem.mem[addr + 0]};
  endfunction

  string memh_path;
  string vcd_path;
  string log_path;
  int log_fd;
  bit log_commits;

  longint unsigned max_cycles;
  longint unsigned expected_mem100;
  longint unsigned expected_a0;
  bit has_expected_mem100;
  bit has_expected_a0;
  logic [31:0] got_mem100;
  int i;

  initial begin
    clk = 1'b0;
    rst = 1'b1;

    boot_pc = 64'h0000_0000_0001_0000;
    boot_sp = 64'h0000_0000_0002_0000;
    max_cycles = 400000;

    if (!$value$plusargs("memh=%s", memh_path)) begin
      memh_path = "janus/programs/test_or.memh";
    end

    void'($value$plusargs("boot_pc=%h", boot_pc));
    void'($value$plusargs("boot_sp=%h", boot_sp));
    void'($value$plusargs("max_cycles=%d", max_cycles));

    has_expected_mem100 = $value$plusargs("expected_mem100=%h", expected_mem100);
    has_expected_a0 = $value$plusargs("expected_a0=%h", expected_a0);

    vcd_path = "janus/generated/janus_bcc_ooo_pyc/tb_janus_bcc_ooo_pyc_sv.vcd";
    log_path = "janus/generated/janus_bcc_ooo_pyc/tb_janus_bcc_ooo_pyc_sv.log";
    void'($value$plusargs("vcd=%s", vcd_path));
    void'($value$plusargs("log=%s", log_path));
    log_commits = $test$plusargs("logcommits");

    if (!$test$plusargs("notrace")) begin
      $display("tb_janus_bcc_ooo_pyc: dumping VCD to %s", vcd_path);
      $dumpfile(vcd_path);
      $dumpvars(0, tb_janus_bcc_ooo_pyc);
    end

    if (!$test$plusargs("nolog")) begin
      log_fd = $fopen(log_path, "w");
      $fdisplay(log_fd, "tb_janus_bcc_ooo_pyc(SV): memh=%s", memh_path);
      $fdisplay(log_fd, "cycle,time,halted,pc,fpc,cycles,rob_count,a0");
    end else begin
      log_fd = 0;
    end

    $display("tb_janus_bcc_ooo_pyc: memh=%s boot_pc=0x%016x max_cycles=%0d", memh_path, boot_pc, max_cycles);

    $readmemh(memh_path, dut.mem.mem);

    repeat (5) @(posedge clk);
    rst = 1'b0;

    i = 0;
    while (i < max_cycles && !halted) begin
      @(posedge clk);
      if (log_fd != 0 && log_commits && commit_fire) begin
        $fdisplay(log_fd,
                  "%0d,%0t,%0b,0x%016x,0x%016x,%0d,%0d,0x%016x",
                  i,
                  $time,
                  halted,
                  pc,
                  fpc,
                  cycles,
                  rob_count,
                  a0);
      end
      i++;
    end

    if (!halted) begin
      $fatal(1, "FAIL: did not halt (pc=0x%016x fpc=0x%016x cycles=%0d)", pc, fpc, cycles);
    end

    got_mem100 = mem_read32(32'h0000_0100);

    if (has_expected_mem100 && got_mem100 !== expected_mem100[31:0]) begin
      $fatal(1, "FAIL: mem[0x100]=0x%08x expected=0x%08x", got_mem100, expected_mem100[31:0]);
    end

    if (has_expected_a0 && a0 !== expected_a0[63:0]) begin
      $fatal(1, "FAIL: a0=0x%016x expected=0x%016x", a0, expected_a0[63:0]);
    end

    if (log_fd != 0) begin
      $fdisplay(log_fd, "PASS: cycles=%0d pc=0x%016x fpc=0x%016x a0=0x%016x mem100=0x%08x", cycles, pc, fpc, a0, got_mem100);
      $fclose(log_fd);
    end

    $display("PASS: cycles=%0d pc=0x%016x a0=0x%016x mem[0x100]=0x%08x", cycles, pc, a0, got_mem100);
    $finish;
  end

endmodule
