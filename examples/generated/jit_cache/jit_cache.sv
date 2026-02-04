`include "pyc_add.sv"
`include "pyc_mux.sv"
`include "pyc_and.sv"
`include "pyc_or.sv"
`include "pyc_xor.sv"
`include "pyc_not.sv"
`include "pyc_reg.sv"
`include "pyc_fifo.sv"

`include "pyc_byte_mem.sv"

module JitCache (
  input logic sys_clk,
  input logic sys_rst,
  input logic req_valid,
  input logic [31:0] req_addr,
  input logic rsp_ready,
  output logic req_ready,
  output logic rsp_valid,
  output logic rsp_hit,
  output logic [31:0] rsp_rdata
);

logic [32:0] v1;
logic [2:0] v2;
logic [2:0] v3;
logic [2:0] v4;
logic [2:0] v5;
logic [2:0] v6;
logic [2:0] v7;
logic [2:0] v8;
logic [2:0] v9;
logic [26:0] v10;
logic v11;
logic [3:0] v12;
logic [31:0] v13;
logic v14;
logic [32:0] v15;
logic [2:0] v16;
logic [2:0] v17;
logic [2:0] v18;
logic [2:0] v19;
logic [2:0] v20;
logic [2:0] v21;
logic [2:0] v22;
logic [2:0] v23;
logic [26:0] v24;
logic v25;
logic [3:0] v26;
logic [31:0] v27;
logic v28;
logic req_valid__jit_cache__L166;
logic [31:0] req_addr__jit_cache__L167;
logic rsp_ready__jit_cache__L168;
logic cache__wvalid0__jit_cache__L172;
logic [31:0] cache__waddr0__jit_cache__L173;
logic [31:0] cache__wdata0__jit_cache__L174;
logic [3:0] cache__wstrb0__jit_cache__L175;
logic cache__req_q__in_valid;
logic [31:0] cache__req_q__in_data;
logic cache__req_q__out_ready;
logic v29;
logic v30;
logic [31:0] v31;
logic cache__rsp_q__in_valid;
logic [32:0] cache__rsp_q__in_data;
logic cache__rsp_q__out_ready;
logic v32;
logic v33;
logic [32:0] v34;
logic v35;
logic cache__rsp_hit__jit_cache__L183;
logic [31:0] v36;
logic [31:0] cache__rsp_rdata__jit_cache__L184;
logic cache__valid__s0__w0__next;
logic v37;
logic cache__valid__s0__w0;
logic cache__valid__s0__w1__next;
logic v38;
logic cache__valid__s0__w1;
logic cache__valid__s1__w0__next;
logic v39;
logic cache__valid__s1__w0;
logic cache__valid__s1__w1__next;
logic v40;
logic cache__valid__s1__w1;
logic cache__valid__s2__w0__next;
logic v41;
logic cache__valid__s2__w0;
logic cache__valid__s2__w1__next;
logic v42;
logic cache__valid__s2__w1;
logic cache__valid__s3__w0__next;
logic v43;
logic cache__valid__s3__w0;
logic cache__valid__s3__w1__next;
logic v44;
logic cache__valid__s3__w1;
logic cache__valid__s4__w0__next;
logic v45;
logic cache__valid__s4__w0;
logic cache__valid__s4__w1__next;
logic v46;
logic cache__valid__s4__w1;
logic cache__valid__s5__w0__next;
logic v47;
logic cache__valid__s5__w0;
logic cache__valid__s5__w1__next;
logic v48;
logic cache__valid__s5__w1;
logic cache__valid__s6__w0__next;
logic v49;
logic cache__valid__s6__w0;
logic cache__valid__s6__w1__next;
logic v50;
logic cache__valid__s6__w1;
logic cache__valid__s7__w0__next;
logic v51;
logic cache__valid__s7__w0;
logic cache__valid__s7__w1__next;
logic v52;
logic cache__valid__s7__w1;
logic [26:0] cache__tag__s0__w0__next;
logic [26:0] v53;
logic [26:0] cache__tag__s0__w0;
logic [26:0] cache__tag__s0__w1__next;
logic [26:0] v54;
logic [26:0] cache__tag__s0__w1;
logic [26:0] cache__tag__s1__w0__next;
logic [26:0] v55;
logic [26:0] cache__tag__s1__w0;
logic [26:0] cache__tag__s1__w1__next;
logic [26:0] v56;
logic [26:0] cache__tag__s1__w1;
logic [26:0] cache__tag__s2__w0__next;
logic [26:0] v57;
logic [26:0] cache__tag__s2__w0;
logic [26:0] cache__tag__s2__w1__next;
logic [26:0] v58;
logic [26:0] cache__tag__s2__w1;
logic [26:0] cache__tag__s3__w0__next;
logic [26:0] v59;
logic [26:0] cache__tag__s3__w0;
logic [26:0] cache__tag__s3__w1__next;
logic [26:0] v60;
logic [26:0] cache__tag__s3__w1;
logic [26:0] cache__tag__s4__w0__next;
logic [26:0] v61;
logic [26:0] cache__tag__s4__w0;
logic [26:0] cache__tag__s4__w1__next;
logic [26:0] v62;
logic [26:0] cache__tag__s4__w1;
logic [26:0] cache__tag__s5__w0__next;
logic [26:0] v63;
logic [26:0] cache__tag__s5__w0;
logic [26:0] cache__tag__s5__w1__next;
logic [26:0] v64;
logic [26:0] cache__tag__s5__w1;
logic [26:0] cache__tag__s6__w0__next;
logic [26:0] v65;
logic [26:0] cache__tag__s6__w0;
logic [26:0] cache__tag__s6__w1__next;
logic [26:0] v66;
logic [26:0] cache__tag__s6__w1;
logic [26:0] cache__tag__s7__w0__next;
logic [26:0] v67;
logic [26:0] cache__tag__s7__w0;
logic [26:0] cache__tag__s7__w1__next;
logic [26:0] v68;
logic [26:0] cache__tag__s7__w1;
logic [31:0] cache__data__s0__w0__next;
logic [31:0] v69;
logic [31:0] cache__data__s0__w0;
logic [31:0] cache__data__s0__w1__next;
logic [31:0] v70;
logic [31:0] cache__data__s0__w1;
logic [31:0] cache__data__s1__w0__next;
logic [31:0] v71;
logic [31:0] cache__data__s1__w0;
logic [31:0] cache__data__s1__w1__next;
logic [31:0] v72;
logic [31:0] cache__data__s1__w1;
logic [31:0] cache__data__s2__w0__next;
logic [31:0] v73;
logic [31:0] cache__data__s2__w0;
logic [31:0] cache__data__s2__w1__next;
logic [31:0] v74;
logic [31:0] cache__data__s2__w1;
logic [31:0] cache__data__s3__w0__next;
logic [31:0] v75;
logic [31:0] cache__data__s3__w0;
logic [31:0] cache__data__s3__w1__next;
logic [31:0] v76;
logic [31:0] cache__data__s3__w1;
logic [31:0] cache__data__s4__w0__next;
logic [31:0] v77;
logic [31:0] cache__data__s4__w0;
logic [31:0] cache__data__s4__w1__next;
logic [31:0] v78;
logic [31:0] cache__data__s4__w1;
logic [31:0] cache__data__s5__w0__next;
logic [31:0] v79;
logic [31:0] cache__data__s5__w0;
logic [31:0] cache__data__s5__w1__next;
logic [31:0] v80;
logic [31:0] cache__data__s5__w1;
logic [31:0] cache__data__s6__w0__next;
logic [31:0] v81;
logic [31:0] cache__data__s6__w0;
logic [31:0] cache__data__s6__w1__next;
logic [31:0] v82;
logic [31:0] cache__data__s6__w1;
logic [31:0] cache__data__s7__w0__next;
logic [31:0] v83;
logic [31:0] cache__data__s7__w0;
logic [31:0] cache__data__s7__w1__next;
logic [31:0] v84;
logic [31:0] cache__data__s7__w1;
logic cache__rr__s0__next;
logic v85;
logic cache__rr__s0;
logic cache__rr__s1__next;
logic v86;
logic cache__rr__s1;
logic cache__rr__s2__next;
logic v87;
logic cache__rr__s2;
logic cache__rr__s3__next;
logic v88;
logic cache__rr__s3;
logic cache__rr__s4__next;
logic v89;
logic cache__rr__s4;
logic cache__rr__s5__next;
logic v90;
logic cache__rr__s5;
logic cache__rr__s6__next;
logic v91;
logic cache__rr__s6;
logic cache__rr__s7__next;
logic v92;
logic cache__rr__s7;
logic v93;
logic cache__req_fire__jit_cache__L194;
logic [31:0] cache__addr__jit_cache__L196;
logic [2:0] v94;
logic [2:0] cache__set_idx__jit_cache__L197;
logic [26:0] v95;
logic [26:0] cache__tag__jit_cache__L198;
logic v96;
logic v97;
logic v98;
logic v99;
logic v100;
logic v101;
logic v102;
logic v103;
logic v104;
logic v105;
logic v106;
logic v107;
logic v108;
logic v109;
logic v110;
logic v111;
logic [26:0] v112;
logic [26:0] v113;
logic [26:0] v114;
logic [26:0] v115;
logic [26:0] v116;
logic [26:0] v117;
logic [26:0] v118;
logic [26:0] v119;
logic [31:0] v120;
logic [31:0] v121;
logic [31:0] v122;
logic [31:0] v123;
logic [31:0] v124;
logic [31:0] v125;
logic [31:0] v126;
logic [31:0] v127;
logic v128;
logic v129;
logic v130;
logic [31:0] v131;
logic v132;
logic v133;
logic v134;
logic v135;
logic v136;
logic v137;
logic v138;
logic v139;
logic [26:0] v140;
logic [26:0] v141;
logic [26:0] v142;
logic [26:0] v143;
logic [26:0] v144;
logic [26:0] v145;
logic [26:0] v146;
logic [26:0] v147;
logic [31:0] v148;
logic [31:0] v149;
logic [31:0] v150;
logic [31:0] v151;
logic [31:0] v152;
logic [31:0] v153;
logic [31:0] v154;
logic [31:0] v155;
logic v156;
logic v157;
logic v158;
logic [31:0] v159;
logic v160;
logic v161;
logic v162;
logic v163;
logic v164;
logic v165;
logic v166;
logic v167;
logic v168;
logic [31:0] v169;
logic cache__hit__jit_cache__L202;
logic [31:0] cache__hit_data__jit_cache__L203;
logic [31:0] v170;
logic [31:0] cache__mem_rdata__jit_cache__L206;
logic v171;
logic v172;
logic v173;
logic cache__miss__jit_cache__L218;
logic v174;
logic v175;
logic v176;
logic v177;
logic v178;
logic v179;
logic v180;
logic v181;
logic v182;
logic cache__repl_way__jit_cache__L219;
logic v183;
logic v184;
logic v185;
logic v186;
logic v187;
logic v188;
logic v189;
logic v190;
logic v191;
logic v192;
logic v193;
logic v194;
logic v195;
logic [26:0] v196;
logic [31:0] v197;
logic v198;
logic v199;
logic v200;
logic v201;
logic v202;
logic v203;
logic [26:0] v204;
logic [31:0] v205;
logic v206;
logic v207;
logic v208;
logic v209;
logic v210;
logic v211;
logic v212;
logic v213;
logic v214;
logic v215;
logic v216;
logic [26:0] v217;
logic [31:0] v218;
logic v219;
logic v220;
logic v221;
logic v222;
logic [26:0] v223;
logic [31:0] v224;
logic v225;
logic v226;
logic v227;
logic v228;
logic v229;
logic v230;
logic v231;
logic v232;
logic v233;
logic v234;
logic v235;
logic [26:0] v236;
logic [31:0] v237;
logic v238;
logic v239;
logic v240;
logic v241;
logic [26:0] v242;
logic [31:0] v243;
logic v244;
logic v245;
logic v246;
logic v247;
logic v248;
logic v249;
logic v250;
logic v251;
logic v252;
logic v253;
logic v254;
logic [26:0] v255;
logic [31:0] v256;
logic v257;
logic v258;
logic v259;
logic v260;
logic [26:0] v261;
logic [31:0] v262;
logic v263;
logic v264;
logic v265;
logic v266;
logic v267;
logic v268;
logic v269;
logic v270;
logic v271;
logic v272;
logic v273;
logic [26:0] v274;
logic [31:0] v275;
logic v276;
logic v277;
logic v278;
logic v279;
logic [26:0] v280;
logic [31:0] v281;
logic v282;
logic v283;
logic v284;
logic v285;
logic v286;
logic v287;
logic v288;
logic v289;
logic v290;
logic v291;
logic v292;
logic [26:0] v293;
logic [31:0] v294;
logic v295;
logic v296;
logic v297;
logic v298;
logic [26:0] v299;
logic [31:0] v300;
logic v301;
logic v302;
logic v303;
logic v304;
logic v305;
logic v306;
logic v307;
logic v308;
logic v309;
logic v310;
logic v311;
logic [26:0] v312;
logic [31:0] v313;
logic v314;
logic v315;
logic v316;
logic v317;
logic [26:0] v318;
logic [31:0] v319;
logic v320;
logic v321;
logic v322;
logic v323;
logic v324;
logic v325;
logic v326;
logic v327;
logic v328;
logic v329;
logic v330;
logic [26:0] v331;
logic [31:0] v332;
logic v333;
logic v334;
logic v335;
logic v336;
logic [26:0] v337;
logic [31:0] v338;
logic [31:0] v339;
logic [31:0] cache__rdata__jit_cache__L232;
logic [32:0] v340;
logic [32:0] v341;
logic [32:0] v342;
logic [32:0] v343;
logic [32:0] v344;
logic [32:0] v345;
logic [32:0] cache__rsp_pkt__jit_cache__L233;

assign v1 = 33'd0;
assign v2 = 3'd7;
assign v3 = 3'd6;
assign v4 = 3'd5;
assign v5 = 3'd4;
assign v6 = 3'd3;
assign v7 = 3'd2;
assign v8 = 3'd1;
assign v9 = 3'd0;
assign v10 = 27'd0;
assign v11 = 1'd1;
assign v12 = 4'd0;
assign v13 = 32'd0;
assign v14 = 1'd0;
assign v15 = v1;
assign v16 = v2;
assign v17 = v3;
assign v18 = v4;
assign v19 = v5;
assign v20 = v6;
assign v21 = v7;
assign v22 = v8;
assign v23 = v9;
assign v24 = v10;
assign v25 = v11;
assign v26 = v12;
assign v27 = v13;
assign v28 = v14;
assign req_valid__jit_cache__L166 = req_valid;
assign req_addr__jit_cache__L167 = req_addr;
assign rsp_ready__jit_cache__L168 = rsp_ready;
assign cache__wvalid0__jit_cache__L172 = v28;
assign cache__waddr0__jit_cache__L173 = v27;
assign cache__wdata0__jit_cache__L174 = v27;
assign cache__wstrb0__jit_cache__L175 = v26;
pyc_fifo #(.WIDTH(32), .DEPTH(2)) v29_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .in_valid(cache__req_q__in_valid),
  .in_ready(v29),
  .in_data(cache__req_q__in_data),
  .out_valid(v30),
  .out_ready(cache__req_q__out_ready),
  .out_data(v31)
);
pyc_fifo #(.WIDTH(33), .DEPTH(2)) v32_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .in_valid(cache__rsp_q__in_valid),
  .in_ready(v32),
  .in_data(cache__rsp_q__in_data),
  .out_valid(v33),
  .out_ready(cache__rsp_q__out_ready),
  .out_data(v34)
);
assign v35 = v34[32];
assign cache__rsp_hit__jit_cache__L183 = v35;
assign v36 = v34[31:0];
assign cache__rsp_rdata__jit_cache__L184 = v36;
pyc_reg #(.WIDTH(1)) v37_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__valid__s0__w0__next),
  .init(v28),
  .q(v37)
);
assign cache__valid__s0__w0 = v37;
pyc_reg #(.WIDTH(1)) v38_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__valid__s0__w1__next),
  .init(v28),
  .q(v38)
);
assign cache__valid__s0__w1 = v38;
pyc_reg #(.WIDTH(1)) v39_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__valid__s1__w0__next),
  .init(v28),
  .q(v39)
);
assign cache__valid__s1__w0 = v39;
pyc_reg #(.WIDTH(1)) v40_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__valid__s1__w1__next),
  .init(v28),
  .q(v40)
);
assign cache__valid__s1__w1 = v40;
pyc_reg #(.WIDTH(1)) v41_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__valid__s2__w0__next),
  .init(v28),
  .q(v41)
);
assign cache__valid__s2__w0 = v41;
pyc_reg #(.WIDTH(1)) v42_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__valid__s2__w1__next),
  .init(v28),
  .q(v42)
);
assign cache__valid__s2__w1 = v42;
pyc_reg #(.WIDTH(1)) v43_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__valid__s3__w0__next),
  .init(v28),
  .q(v43)
);
assign cache__valid__s3__w0 = v43;
pyc_reg #(.WIDTH(1)) v44_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__valid__s3__w1__next),
  .init(v28),
  .q(v44)
);
assign cache__valid__s3__w1 = v44;
pyc_reg #(.WIDTH(1)) v45_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__valid__s4__w0__next),
  .init(v28),
  .q(v45)
);
assign cache__valid__s4__w0 = v45;
pyc_reg #(.WIDTH(1)) v46_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__valid__s4__w1__next),
  .init(v28),
  .q(v46)
);
assign cache__valid__s4__w1 = v46;
pyc_reg #(.WIDTH(1)) v47_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__valid__s5__w0__next),
  .init(v28),
  .q(v47)
);
assign cache__valid__s5__w0 = v47;
pyc_reg #(.WIDTH(1)) v48_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__valid__s5__w1__next),
  .init(v28),
  .q(v48)
);
assign cache__valid__s5__w1 = v48;
pyc_reg #(.WIDTH(1)) v49_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__valid__s6__w0__next),
  .init(v28),
  .q(v49)
);
assign cache__valid__s6__w0 = v49;
pyc_reg #(.WIDTH(1)) v50_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__valid__s6__w1__next),
  .init(v28),
  .q(v50)
);
assign cache__valid__s6__w1 = v50;
pyc_reg #(.WIDTH(1)) v51_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__valid__s7__w0__next),
  .init(v28),
  .q(v51)
);
assign cache__valid__s7__w0 = v51;
pyc_reg #(.WIDTH(1)) v52_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__valid__s7__w1__next),
  .init(v28),
  .q(v52)
);
assign cache__valid__s7__w1 = v52;
pyc_reg #(.WIDTH(27)) v53_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__tag__s0__w0__next),
  .init(v24),
  .q(v53)
);
assign cache__tag__s0__w0 = v53;
pyc_reg #(.WIDTH(27)) v54_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__tag__s0__w1__next),
  .init(v24),
  .q(v54)
);
assign cache__tag__s0__w1 = v54;
pyc_reg #(.WIDTH(27)) v55_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__tag__s1__w0__next),
  .init(v24),
  .q(v55)
);
assign cache__tag__s1__w0 = v55;
pyc_reg #(.WIDTH(27)) v56_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__tag__s1__w1__next),
  .init(v24),
  .q(v56)
);
assign cache__tag__s1__w1 = v56;
pyc_reg #(.WIDTH(27)) v57_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__tag__s2__w0__next),
  .init(v24),
  .q(v57)
);
assign cache__tag__s2__w0 = v57;
pyc_reg #(.WIDTH(27)) v58_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__tag__s2__w1__next),
  .init(v24),
  .q(v58)
);
assign cache__tag__s2__w1 = v58;
pyc_reg #(.WIDTH(27)) v59_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__tag__s3__w0__next),
  .init(v24),
  .q(v59)
);
assign cache__tag__s3__w0 = v59;
pyc_reg #(.WIDTH(27)) v60_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__tag__s3__w1__next),
  .init(v24),
  .q(v60)
);
assign cache__tag__s3__w1 = v60;
pyc_reg #(.WIDTH(27)) v61_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__tag__s4__w0__next),
  .init(v24),
  .q(v61)
);
assign cache__tag__s4__w0 = v61;
pyc_reg #(.WIDTH(27)) v62_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__tag__s4__w1__next),
  .init(v24),
  .q(v62)
);
assign cache__tag__s4__w1 = v62;
pyc_reg #(.WIDTH(27)) v63_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__tag__s5__w0__next),
  .init(v24),
  .q(v63)
);
assign cache__tag__s5__w0 = v63;
pyc_reg #(.WIDTH(27)) v64_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__tag__s5__w1__next),
  .init(v24),
  .q(v64)
);
assign cache__tag__s5__w1 = v64;
pyc_reg #(.WIDTH(27)) v65_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__tag__s6__w0__next),
  .init(v24),
  .q(v65)
);
assign cache__tag__s6__w0 = v65;
pyc_reg #(.WIDTH(27)) v66_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__tag__s6__w1__next),
  .init(v24),
  .q(v66)
);
assign cache__tag__s6__w1 = v66;
pyc_reg #(.WIDTH(27)) v67_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__tag__s7__w0__next),
  .init(v24),
  .q(v67)
);
assign cache__tag__s7__w0 = v67;
pyc_reg #(.WIDTH(27)) v68_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__tag__s7__w1__next),
  .init(v24),
  .q(v68)
);
assign cache__tag__s7__w1 = v68;
pyc_reg #(.WIDTH(32)) v69_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__data__s0__w0__next),
  .init(v27),
  .q(v69)
);
assign cache__data__s0__w0 = v69;
pyc_reg #(.WIDTH(32)) v70_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__data__s0__w1__next),
  .init(v27),
  .q(v70)
);
assign cache__data__s0__w1 = v70;
pyc_reg #(.WIDTH(32)) v71_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__data__s1__w0__next),
  .init(v27),
  .q(v71)
);
assign cache__data__s1__w0 = v71;
pyc_reg #(.WIDTH(32)) v72_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__data__s1__w1__next),
  .init(v27),
  .q(v72)
);
assign cache__data__s1__w1 = v72;
pyc_reg #(.WIDTH(32)) v73_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__data__s2__w0__next),
  .init(v27),
  .q(v73)
);
assign cache__data__s2__w0 = v73;
pyc_reg #(.WIDTH(32)) v74_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__data__s2__w1__next),
  .init(v27),
  .q(v74)
);
assign cache__data__s2__w1 = v74;
pyc_reg #(.WIDTH(32)) v75_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__data__s3__w0__next),
  .init(v27),
  .q(v75)
);
assign cache__data__s3__w0 = v75;
pyc_reg #(.WIDTH(32)) v76_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__data__s3__w1__next),
  .init(v27),
  .q(v76)
);
assign cache__data__s3__w1 = v76;
pyc_reg #(.WIDTH(32)) v77_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__data__s4__w0__next),
  .init(v27),
  .q(v77)
);
assign cache__data__s4__w0 = v77;
pyc_reg #(.WIDTH(32)) v78_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__data__s4__w1__next),
  .init(v27),
  .q(v78)
);
assign cache__data__s4__w1 = v78;
pyc_reg #(.WIDTH(32)) v79_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__data__s5__w0__next),
  .init(v27),
  .q(v79)
);
assign cache__data__s5__w0 = v79;
pyc_reg #(.WIDTH(32)) v80_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__data__s5__w1__next),
  .init(v27),
  .q(v80)
);
assign cache__data__s5__w1 = v80;
pyc_reg #(.WIDTH(32)) v81_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__data__s6__w0__next),
  .init(v27),
  .q(v81)
);
assign cache__data__s6__w0 = v81;
pyc_reg #(.WIDTH(32)) v82_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__data__s6__w1__next),
  .init(v27),
  .q(v82)
);
assign cache__data__s6__w1 = v82;
pyc_reg #(.WIDTH(32)) v83_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__data__s7__w0__next),
  .init(v27),
  .q(v83)
);
assign cache__data__s7__w0 = v83;
pyc_reg #(.WIDTH(32)) v84_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__data__s7__w1__next),
  .init(v27),
  .q(v84)
);
assign cache__data__s7__w1 = v84;
pyc_reg #(.WIDTH(1)) v85_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__rr__s0__next),
  .init(v28),
  .q(v85)
);
assign cache__rr__s0 = v85;
pyc_reg #(.WIDTH(1)) v86_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__rr__s1__next),
  .init(v28),
  .q(v86)
);
assign cache__rr__s1 = v86;
pyc_reg #(.WIDTH(1)) v87_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__rr__s2__next),
  .init(v28),
  .q(v87)
);
assign cache__rr__s2 = v87;
pyc_reg #(.WIDTH(1)) v88_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__rr__s3__next),
  .init(v28),
  .q(v88)
);
assign cache__rr__s3 = v88;
pyc_reg #(.WIDTH(1)) v89_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__rr__s4__next),
  .init(v28),
  .q(v89)
);
assign cache__rr__s4 = v89;
pyc_reg #(.WIDTH(1)) v90_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__rr__s5__next),
  .init(v28),
  .q(v90)
);
assign cache__rr__s5 = v90;
pyc_reg #(.WIDTH(1)) v91_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__rr__s6__next),
  .init(v28),
  .q(v91)
);
assign cache__rr__s6 = v91;
pyc_reg #(.WIDTH(1)) v92_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v25),
  .d(cache__rr__s7__next),
  .init(v28),
  .q(v92)
);
assign cache__rr__s7 = v92;
pyc_and #(.WIDTH(1)) v93_inst (
  .a(v30),
  .b(v32),
  .y(v93)
);
assign cache__req_fire__jit_cache__L194 = v93;
assign cache__addr__jit_cache__L196 = v31;
assign v94 = cache__addr__jit_cache__L196[4:2];
assign cache__set_idx__jit_cache__L197 = v94;
assign v95 = cache__addr__jit_cache__L196[31:5];
assign cache__tag__jit_cache__L198 = v95;
assign v96 = (cache__set_idx__jit_cache__L197 == v23);
assign v97 = (v96 ? cache__valid__s0__w0 : v28);
assign v98 = (cache__set_idx__jit_cache__L197 == v22);
assign v99 = (v98 ? cache__valid__s1__w0 : v97);
assign v100 = (cache__set_idx__jit_cache__L197 == v21);
assign v101 = (v100 ? cache__valid__s2__w0 : v99);
assign v102 = (cache__set_idx__jit_cache__L197 == v20);
assign v103 = (v102 ? cache__valid__s3__w0 : v101);
assign v104 = (cache__set_idx__jit_cache__L197 == v19);
assign v105 = (v104 ? cache__valid__s4__w0 : v103);
assign v106 = (cache__set_idx__jit_cache__L197 == v18);
assign v107 = (v106 ? cache__valid__s5__w0 : v105);
assign v108 = (cache__set_idx__jit_cache__L197 == v17);
assign v109 = (v108 ? cache__valid__s6__w0 : v107);
assign v110 = (cache__set_idx__jit_cache__L197 == v16);
assign v111 = (v110 ? cache__valid__s7__w0 : v109);
assign v112 = (v96 ? cache__tag__s0__w0 : v24);
assign v113 = (v98 ? cache__tag__s1__w0 : v112);
assign v114 = (v100 ? cache__tag__s2__w0 : v113);
assign v115 = (v102 ? cache__tag__s3__w0 : v114);
assign v116 = (v104 ? cache__tag__s4__w0 : v115);
assign v117 = (v106 ? cache__tag__s5__w0 : v116);
assign v118 = (v108 ? cache__tag__s6__w0 : v117);
assign v119 = (v110 ? cache__tag__s7__w0 : v118);
assign v120 = (v96 ? cache__data__s0__w0 : v27);
assign v121 = (v98 ? cache__data__s1__w0 : v120);
assign v122 = (v100 ? cache__data__s2__w0 : v121);
assign v123 = (v102 ? cache__data__s3__w0 : v122);
assign v124 = (v104 ? cache__data__s4__w0 : v123);
assign v125 = (v106 ? cache__data__s5__w0 : v124);
assign v126 = (v108 ? cache__data__s6__w0 : v125);
assign v127 = (v110 ? cache__data__s7__w0 : v126);
assign v128 = (v119 == cache__tag__jit_cache__L198);
assign v129 = (v111 & v128);
assign v130 = (v28 | v129);
assign v131 = (v129 ? v127 : v27);
assign v132 = (v96 ? cache__valid__s0__w1 : v28);
assign v133 = (v98 ? cache__valid__s1__w1 : v132);
assign v134 = (v100 ? cache__valid__s2__w1 : v133);
assign v135 = (v102 ? cache__valid__s3__w1 : v134);
assign v136 = (v104 ? cache__valid__s4__w1 : v135);
assign v137 = (v106 ? cache__valid__s5__w1 : v136);
assign v138 = (v108 ? cache__valid__s6__w1 : v137);
assign v139 = (v110 ? cache__valid__s7__w1 : v138);
assign v140 = (v96 ? cache__tag__s0__w1 : v24);
assign v141 = (v98 ? cache__tag__s1__w1 : v140);
assign v142 = (v100 ? cache__tag__s2__w1 : v141);
assign v143 = (v102 ? cache__tag__s3__w1 : v142);
assign v144 = (v104 ? cache__tag__s4__w1 : v143);
assign v145 = (v106 ? cache__tag__s5__w1 : v144);
assign v146 = (v108 ? cache__tag__s6__w1 : v145);
assign v147 = (v110 ? cache__tag__s7__w1 : v146);
assign v148 = (v96 ? cache__data__s0__w1 : v27);
assign v149 = (v98 ? cache__data__s1__w1 : v148);
assign v150 = (v100 ? cache__data__s2__w1 : v149);
assign v151 = (v102 ? cache__data__s3__w1 : v150);
assign v152 = (v104 ? cache__data__s4__w1 : v151);
assign v153 = (v106 ? cache__data__s5__w1 : v152);
assign v154 = (v108 ? cache__data__s6__w1 : v153);
assign v155 = (v110 ? cache__data__s7__w1 : v154);
assign v156 = (v147 == cache__tag__jit_cache__L198);
assign v157 = (v139 & v156);
assign v158 = (v130 | v157);
assign v159 = (v157 ? v155 : v131);
assign v160 = v96;
assign v161 = v98;
assign v162 = v100;
assign v163 = v102;
assign v164 = v104;
assign v165 = v106;
assign v166 = v108;
assign v167 = v110;
assign v168 = v158;
assign v169 = v159;
assign cache__hit__jit_cache__L202 = v168;
assign cache__hit_data__jit_cache__L203 = v169;
pyc_byte_mem #(.ADDR_WIDTH(32), .DATA_WIDTH(32), .DEPTH(4096)) main_mem (
  .clk(sys_clk),
  .rst(sys_rst),
  .raddr(cache__addr__jit_cache__L196),
  .rdata(v170),
  .wvalid(cache__wvalid0__jit_cache__L172),
  .waddr(cache__waddr0__jit_cache__L173),
  .wdata(cache__wdata0__jit_cache__L174),
  .wstrb(cache__wstrb0__jit_cache__L175)
);
assign cache__mem_rdata__jit_cache__L206 = v170;
assign v171 = (~cache__hit__jit_cache__L202);
assign v172 = (cache__req_fire__jit_cache__L194 & v171);
assign v173 = v172;
assign cache__miss__jit_cache__L218 = v173;
assign v174 = (v160 ? cache__rr__s0 : v28);
assign v175 = (v161 ? cache__rr__s1 : v174);
assign v176 = (v162 ? cache__rr__s2 : v175);
assign v177 = (v163 ? cache__rr__s3 : v176);
assign v178 = (v164 ? cache__rr__s4 : v177);
assign v179 = (v165 ? cache__rr__s5 : v178);
assign v180 = (v166 ? cache__rr__s6 : v179);
assign v181 = (v167 ? cache__rr__s7 : v180);
assign v182 = v181;
assign cache__repl_way__jit_cache__L219 = v182;
assign v183 = (cache__miss__jit_cache__L218 & v160);
assign v184 = (cache__rr__s0 + v25);
assign v185 = (cache__rr__s0 == v25);
assign v186 = (v185 ? v28 : v184);
assign v187 = (v183 ? v186 : cache__rr__s0);
assign v188 = v183;
assign v189 = v187;
assign cache__rr__s0__next = v189;
assign v190 = (cache__repl_way__jit_cache__L219 == v28);
assign v191 = (v188 & v190);
assign v192 = (v191 ? v25 : cache__valid__s0__w0);
assign v193 = v190;
assign v194 = v191;
assign v195 = v192;
assign cache__valid__s0__w0__next = v195;
pyc_mux #(.WIDTH(27)) v196_inst (
  .sel(v194),
  .a(cache__tag__jit_cache__L198),
  .b(cache__tag__s0__w0),
  .y(v196)
);
assign cache__tag__s0__w0__next = v196;
pyc_mux #(.WIDTH(32)) v197_inst (
  .sel(v194),
  .a(cache__mem_rdata__jit_cache__L206),
  .b(cache__data__s0__w0),
  .y(v197)
);
assign cache__data__s0__w0__next = v197;
assign v198 = (cache__repl_way__jit_cache__L219 == v25);
assign v199 = (v188 & v198);
assign v200 = (v199 ? v25 : cache__valid__s0__w1);
assign v201 = v198;
assign v202 = v199;
assign v203 = v200;
assign cache__valid__s0__w1__next = v203;
pyc_mux #(.WIDTH(27)) v204_inst (
  .sel(v202),
  .a(cache__tag__jit_cache__L198),
  .b(cache__tag__s0__w1),
  .y(v204)
);
assign cache__tag__s0__w1__next = v204;
pyc_mux #(.WIDTH(32)) v205_inst (
  .sel(v202),
  .a(cache__mem_rdata__jit_cache__L206),
  .b(cache__data__s0__w1),
  .y(v205)
);
assign cache__data__s0__w1__next = v205;
assign v206 = (cache__miss__jit_cache__L218 & v161);
assign v207 = (cache__rr__s1 + v25);
assign v208 = (cache__rr__s1 == v25);
assign v209 = (v208 ? v28 : v207);
assign v210 = (v206 ? v209 : cache__rr__s1);
assign v211 = v206;
assign v212 = v210;
assign cache__rr__s1__next = v212;
assign v213 = (v211 & v193);
assign v214 = (v213 ? v25 : cache__valid__s1__w0);
assign v215 = v213;
assign v216 = v214;
assign cache__valid__s1__w0__next = v216;
pyc_mux #(.WIDTH(27)) v217_inst (
  .sel(v215),
  .a(cache__tag__jit_cache__L198),
  .b(cache__tag__s1__w0),
  .y(v217)
);
assign cache__tag__s1__w0__next = v217;
pyc_mux #(.WIDTH(32)) v218_inst (
  .sel(v215),
  .a(cache__mem_rdata__jit_cache__L206),
  .b(cache__data__s1__w0),
  .y(v218)
);
assign cache__data__s1__w0__next = v218;
assign v219 = (v211 & v201);
assign v220 = (v219 ? v25 : cache__valid__s1__w1);
assign v221 = v219;
assign v222 = v220;
assign cache__valid__s1__w1__next = v222;
pyc_mux #(.WIDTH(27)) v223_inst (
  .sel(v221),
  .a(cache__tag__jit_cache__L198),
  .b(cache__tag__s1__w1),
  .y(v223)
);
assign cache__tag__s1__w1__next = v223;
pyc_mux #(.WIDTH(32)) v224_inst (
  .sel(v221),
  .a(cache__mem_rdata__jit_cache__L206),
  .b(cache__data__s1__w1),
  .y(v224)
);
assign cache__data__s1__w1__next = v224;
assign v225 = (cache__miss__jit_cache__L218 & v162);
assign v226 = (cache__rr__s2 + v25);
assign v227 = (cache__rr__s2 == v25);
assign v228 = (v227 ? v28 : v226);
assign v229 = (v225 ? v228 : cache__rr__s2);
assign v230 = v225;
assign v231 = v229;
assign cache__rr__s2__next = v231;
assign v232 = (v230 & v193);
assign v233 = (v232 ? v25 : cache__valid__s2__w0);
assign v234 = v232;
assign v235 = v233;
assign cache__valid__s2__w0__next = v235;
pyc_mux #(.WIDTH(27)) v236_inst (
  .sel(v234),
  .a(cache__tag__jit_cache__L198),
  .b(cache__tag__s2__w0),
  .y(v236)
);
assign cache__tag__s2__w0__next = v236;
pyc_mux #(.WIDTH(32)) v237_inst (
  .sel(v234),
  .a(cache__mem_rdata__jit_cache__L206),
  .b(cache__data__s2__w0),
  .y(v237)
);
assign cache__data__s2__w0__next = v237;
assign v238 = (v230 & v201);
assign v239 = (v238 ? v25 : cache__valid__s2__w1);
assign v240 = v238;
assign v241 = v239;
assign cache__valid__s2__w1__next = v241;
pyc_mux #(.WIDTH(27)) v242_inst (
  .sel(v240),
  .a(cache__tag__jit_cache__L198),
  .b(cache__tag__s2__w1),
  .y(v242)
);
assign cache__tag__s2__w1__next = v242;
pyc_mux #(.WIDTH(32)) v243_inst (
  .sel(v240),
  .a(cache__mem_rdata__jit_cache__L206),
  .b(cache__data__s2__w1),
  .y(v243)
);
assign cache__data__s2__w1__next = v243;
assign v244 = (cache__miss__jit_cache__L218 & v163);
assign v245 = (cache__rr__s3 + v25);
assign v246 = (cache__rr__s3 == v25);
assign v247 = (v246 ? v28 : v245);
assign v248 = (v244 ? v247 : cache__rr__s3);
assign v249 = v244;
assign v250 = v248;
assign cache__rr__s3__next = v250;
assign v251 = (v249 & v193);
assign v252 = (v251 ? v25 : cache__valid__s3__w0);
assign v253 = v251;
assign v254 = v252;
assign cache__valid__s3__w0__next = v254;
pyc_mux #(.WIDTH(27)) v255_inst (
  .sel(v253),
  .a(cache__tag__jit_cache__L198),
  .b(cache__tag__s3__w0),
  .y(v255)
);
assign cache__tag__s3__w0__next = v255;
pyc_mux #(.WIDTH(32)) v256_inst (
  .sel(v253),
  .a(cache__mem_rdata__jit_cache__L206),
  .b(cache__data__s3__w0),
  .y(v256)
);
assign cache__data__s3__w0__next = v256;
assign v257 = (v249 & v201);
assign v258 = (v257 ? v25 : cache__valid__s3__w1);
assign v259 = v257;
assign v260 = v258;
assign cache__valid__s3__w1__next = v260;
pyc_mux #(.WIDTH(27)) v261_inst (
  .sel(v259),
  .a(cache__tag__jit_cache__L198),
  .b(cache__tag__s3__w1),
  .y(v261)
);
assign cache__tag__s3__w1__next = v261;
pyc_mux #(.WIDTH(32)) v262_inst (
  .sel(v259),
  .a(cache__mem_rdata__jit_cache__L206),
  .b(cache__data__s3__w1),
  .y(v262)
);
assign cache__data__s3__w1__next = v262;
assign v263 = (cache__miss__jit_cache__L218 & v164);
assign v264 = (cache__rr__s4 + v25);
assign v265 = (cache__rr__s4 == v25);
assign v266 = (v265 ? v28 : v264);
assign v267 = (v263 ? v266 : cache__rr__s4);
assign v268 = v263;
assign v269 = v267;
assign cache__rr__s4__next = v269;
assign v270 = (v268 & v193);
assign v271 = (v270 ? v25 : cache__valid__s4__w0);
assign v272 = v270;
assign v273 = v271;
assign cache__valid__s4__w0__next = v273;
pyc_mux #(.WIDTH(27)) v274_inst (
  .sel(v272),
  .a(cache__tag__jit_cache__L198),
  .b(cache__tag__s4__w0),
  .y(v274)
);
assign cache__tag__s4__w0__next = v274;
pyc_mux #(.WIDTH(32)) v275_inst (
  .sel(v272),
  .a(cache__mem_rdata__jit_cache__L206),
  .b(cache__data__s4__w0),
  .y(v275)
);
assign cache__data__s4__w0__next = v275;
assign v276 = (v268 & v201);
assign v277 = (v276 ? v25 : cache__valid__s4__w1);
assign v278 = v276;
assign v279 = v277;
assign cache__valid__s4__w1__next = v279;
pyc_mux #(.WIDTH(27)) v280_inst (
  .sel(v278),
  .a(cache__tag__jit_cache__L198),
  .b(cache__tag__s4__w1),
  .y(v280)
);
assign cache__tag__s4__w1__next = v280;
pyc_mux #(.WIDTH(32)) v281_inst (
  .sel(v278),
  .a(cache__mem_rdata__jit_cache__L206),
  .b(cache__data__s4__w1),
  .y(v281)
);
assign cache__data__s4__w1__next = v281;
assign v282 = (cache__miss__jit_cache__L218 & v165);
assign v283 = (cache__rr__s5 + v25);
assign v284 = (cache__rr__s5 == v25);
assign v285 = (v284 ? v28 : v283);
assign v286 = (v282 ? v285 : cache__rr__s5);
assign v287 = v282;
assign v288 = v286;
assign cache__rr__s5__next = v288;
assign v289 = (v287 & v193);
assign v290 = (v289 ? v25 : cache__valid__s5__w0);
assign v291 = v289;
assign v292 = v290;
assign cache__valid__s5__w0__next = v292;
pyc_mux #(.WIDTH(27)) v293_inst (
  .sel(v291),
  .a(cache__tag__jit_cache__L198),
  .b(cache__tag__s5__w0),
  .y(v293)
);
assign cache__tag__s5__w0__next = v293;
pyc_mux #(.WIDTH(32)) v294_inst (
  .sel(v291),
  .a(cache__mem_rdata__jit_cache__L206),
  .b(cache__data__s5__w0),
  .y(v294)
);
assign cache__data__s5__w0__next = v294;
assign v295 = (v287 & v201);
assign v296 = (v295 ? v25 : cache__valid__s5__w1);
assign v297 = v295;
assign v298 = v296;
assign cache__valid__s5__w1__next = v298;
pyc_mux #(.WIDTH(27)) v299_inst (
  .sel(v297),
  .a(cache__tag__jit_cache__L198),
  .b(cache__tag__s5__w1),
  .y(v299)
);
assign cache__tag__s5__w1__next = v299;
pyc_mux #(.WIDTH(32)) v300_inst (
  .sel(v297),
  .a(cache__mem_rdata__jit_cache__L206),
  .b(cache__data__s5__w1),
  .y(v300)
);
assign cache__data__s5__w1__next = v300;
assign v301 = (cache__miss__jit_cache__L218 & v166);
assign v302 = (cache__rr__s6 + v25);
assign v303 = (cache__rr__s6 == v25);
assign v304 = (v303 ? v28 : v302);
assign v305 = (v301 ? v304 : cache__rr__s6);
assign v306 = v301;
assign v307 = v305;
assign cache__rr__s6__next = v307;
assign v308 = (v306 & v193);
assign v309 = (v308 ? v25 : cache__valid__s6__w0);
assign v310 = v308;
assign v311 = v309;
assign cache__valid__s6__w0__next = v311;
pyc_mux #(.WIDTH(27)) v312_inst (
  .sel(v310),
  .a(cache__tag__jit_cache__L198),
  .b(cache__tag__s6__w0),
  .y(v312)
);
assign cache__tag__s6__w0__next = v312;
pyc_mux #(.WIDTH(32)) v313_inst (
  .sel(v310),
  .a(cache__mem_rdata__jit_cache__L206),
  .b(cache__data__s6__w0),
  .y(v313)
);
assign cache__data__s6__w0__next = v313;
assign v314 = (v306 & v201);
assign v315 = (v314 ? v25 : cache__valid__s6__w1);
assign v316 = v314;
assign v317 = v315;
assign cache__valid__s6__w1__next = v317;
pyc_mux #(.WIDTH(27)) v318_inst (
  .sel(v316),
  .a(cache__tag__jit_cache__L198),
  .b(cache__tag__s6__w1),
  .y(v318)
);
assign cache__tag__s6__w1__next = v318;
pyc_mux #(.WIDTH(32)) v319_inst (
  .sel(v316),
  .a(cache__mem_rdata__jit_cache__L206),
  .b(cache__data__s6__w1),
  .y(v319)
);
assign cache__data__s6__w1__next = v319;
assign v320 = (cache__miss__jit_cache__L218 & v167);
assign v321 = (cache__rr__s7 + v25);
assign v322 = (cache__rr__s7 == v25);
assign v323 = (v322 ? v28 : v321);
assign v324 = (v320 ? v323 : cache__rr__s7);
assign v325 = v320;
assign v326 = v324;
assign cache__rr__s7__next = v326;
assign v327 = (v325 & v193);
assign v328 = (v327 ? v25 : cache__valid__s7__w0);
assign v329 = v327;
assign v330 = v328;
assign cache__valid__s7__w0__next = v330;
pyc_mux #(.WIDTH(27)) v331_inst (
  .sel(v329),
  .a(cache__tag__jit_cache__L198),
  .b(cache__tag__s7__w0),
  .y(v331)
);
assign cache__tag__s7__w0__next = v331;
pyc_mux #(.WIDTH(32)) v332_inst (
  .sel(v329),
  .a(cache__mem_rdata__jit_cache__L206),
  .b(cache__data__s7__w0),
  .y(v332)
);
assign cache__data__s7__w0__next = v332;
assign v333 = (v325 & v201);
assign v334 = (v333 ? v25 : cache__valid__s7__w1);
assign v335 = v333;
assign v336 = v334;
assign cache__valid__s7__w1__next = v336;
pyc_mux #(.WIDTH(27)) v337_inst (
  .sel(v335),
  .a(cache__tag__jit_cache__L198),
  .b(cache__tag__s7__w1),
  .y(v337)
);
assign cache__tag__s7__w1__next = v337;
pyc_mux #(.WIDTH(32)) v338_inst (
  .sel(v335),
  .a(cache__mem_rdata__jit_cache__L206),
  .b(cache__data__s7__w1),
  .y(v338)
);
assign cache__data__s7__w1__next = v338;
pyc_mux #(.WIDTH(32)) v339_inst (
  .sel(cache__hit__jit_cache__L202),
  .a(cache__hit_data__jit_cache__L203),
  .b(cache__mem_rdata__jit_cache__L206),
  .y(v339)
);
assign cache__rdata__jit_cache__L232 = v339;
assign v340 = {{1{1'b0}}, cache__rdata__jit_cache__L232};
assign v341 = (v15 | v340);
assign v342 = {{32{1'b0}}, cache__hit__jit_cache__L202};
assign v343 = (v342 << 32);
assign v344 = (v341 | v343);
assign v345 = v344;
assign cache__rsp_pkt__jit_cache__L233 = v345;
assign cache__req_q__in_valid = req_valid__jit_cache__L166;
assign cache__req_q__in_data = req_addr__jit_cache__L167;
assign cache__req_q__out_ready = v32;
assign cache__rsp_q__in_valid = cache__req_fire__jit_cache__L194;
assign cache__rsp_q__in_data = cache__rsp_pkt__jit_cache__L233;
assign cache__rsp_q__out_ready = rsp_ready__jit_cache__L168;
assign req_ready = v29;
assign rsp_valid = v33;
assign rsp_hit = cache__rsp_hit__jit_cache__L183;
assign rsp_rdata = cache__rsp_rdata__jit_cache__L184;

endmodule

