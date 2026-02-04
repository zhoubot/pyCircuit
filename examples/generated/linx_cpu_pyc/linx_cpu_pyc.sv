`include "pyc_add.sv"
`include "pyc_mux.sv"
`include "pyc_and.sv"
`include "pyc_or.sv"
`include "pyc_xor.sv"
`include "pyc_not.sv"
`include "pyc_reg.sv"
`include "pyc_fifo.sv"

`include "pyc_byte_mem.sv"

module linx_cpu_pyc (
  input logic clk,
  input logic rst,
  input logic [63:0] boot_pc,
  input logic [63:0] boot_sp,
  output logic halted,
  output logic [63:0] pc,
  output logic [2:0] stage,
  output logic [63:0] cycles,
  output logic [63:0] a0,
  output logic [63:0] a1,
  output logic [63:0] ra,
  output logic [63:0] sp,
  output logic [1:0] br_kind,
  output logic [63:0] if_window,
  output logic [5:0] wb_op,
  output logic [5:0] wb_regdst,
  output logic [63:0] wb_value,
  output logic commit_cond,
  output logic [63:0] commit_tgt
);

logic [2:0] v1;
logic [1:0] v2;
logic [1:0] v3;
logic [1:0] v4;
logic [63:0] v5;
logic [5:0] v6;
logic [5:0] v7;
logic [5:0] v8;
logic [5:0] v9;
logic [5:0] v10;
logic [2:0] v11;
logic [5:0] v12;
logic [47:0] v13;
logic [47:0] v14;
logic [5:0] v15;
logic [31:0] v16;
logic [31:0] v17;
logic [5:0] v18;
logic [31:0] v19;
logic [31:0] v20;
logic [31:0] v21;
logic [31:0] v22;
logic [5:0] v23;
logic [31:0] v24;
logic [31:0] v25;
logic [5:0] v26;
logic [31:0] v27;
logic [5:0] v28;
logic [31:0] v29;
logic [5:0] v30;
logic [31:0] v31;
logic [5:0] v32;
logic [31:0] v33;
logic [5:0] v34;
logic [31:0] v35;
logic [31:0] v36;
logic [5:0] v37;
logic [31:0] v38;
logic [5:0] v39;
logic [31:0] v40;
logic [5:0] v41;
logic [31:0] v42;
logic [5:0] v43;
logic [31:0] v44;
logic [5:0] v45;
logic [31:0] v46;
logic [31:0] v47;
logic [5:0] v48;
logic [15:0] v49;
logic [5:0] v50;
logic [15:0] v51;
logic [15:0] v52;
logic [5:0] v53;
logic [15:0] v54;
logic [15:0] v55;
logic [15:0] v56;
logic [5:0] v57;
logic [15:0] v58;
logic [5:0] v59;
logic [15:0] v60;
logic [5:0] v61;
logic [15:0] v62;
logic [5:0] v63;
logic [5:0] v64;
logic [15:0] v65;
logic [5:0] v66;
logic [5:0] v67;
logic [15:0] v68;
logic [15:0] v69;
logic [5:0] v70;
logic [15:0] v71;
logic [15:0] v72;
logic [3:0] v73;
logic [7:0] v74;
logic [7:0] v75;
logic [5:0] v76;
logic [2:0] v77;
logic [2:0] v78;
logic [2:0] v79;
logic [2:0] v80;
logic [5:0] v81;
logic [1:0] v82;
logic [63:0] v83;
logic [63:0] v84;
logic [7:0] v85;
logic [5:0] v86;
logic [2:0] v87;
logic v88;
logic v89;
logic [2:0] v90;
logic [1:0] v91;
logic [1:0] v92;
logic [1:0] v93;
logic [63:0] v94;
logic [5:0] v95;
logic [5:0] v96;
logic [5:0] v97;
logic [5:0] v98;
logic [5:0] v99;
logic [2:0] v100;
logic [5:0] v101;
logic [47:0] v102;
logic [47:0] v103;
logic [5:0] v104;
logic [31:0] v105;
logic [31:0] v106;
logic [5:0] v107;
logic [31:0] v108;
logic [31:0] v109;
logic [31:0] v110;
logic [31:0] v111;
logic [5:0] v112;
logic [31:0] v113;
logic [31:0] v114;
logic [5:0] v115;
logic [31:0] v116;
logic [5:0] v117;
logic [31:0] v118;
logic [5:0] v119;
logic [31:0] v120;
logic [5:0] v121;
logic [31:0] v122;
logic [5:0] v123;
logic [31:0] v124;
logic [31:0] v125;
logic [5:0] v126;
logic [31:0] v127;
logic [5:0] v128;
logic [31:0] v129;
logic [5:0] v130;
logic [31:0] v131;
logic [5:0] v132;
logic [31:0] v133;
logic [5:0] v134;
logic [31:0] v135;
logic [31:0] v136;
logic [5:0] v137;
logic [15:0] v138;
logic [5:0] v139;
logic [15:0] v140;
logic [15:0] v141;
logic [5:0] v142;
logic [15:0] v143;
logic [15:0] v144;
logic [15:0] v145;
logic [5:0] v146;
logic [15:0] v147;
logic [5:0] v148;
logic [15:0] v149;
logic [5:0] v150;
logic [15:0] v151;
logic [5:0] v152;
logic [5:0] v153;
logic [15:0] v154;
logic [5:0] v155;
logic [5:0] v156;
logic [15:0] v157;
logic [15:0] v158;
logic [5:0] v159;
logic [15:0] v160;
logic [15:0] v161;
logic [3:0] v162;
logic [7:0] v163;
logic [7:0] v164;
logic [5:0] v165;
logic [2:0] v166;
logic [2:0] v167;
logic [2:0] v168;
logic [2:0] v169;
logic [5:0] v170;
logic [1:0] v171;
logic [63:0] v172;
logic [63:0] v173;
logic [7:0] v174;
logic [5:0] v175;
logic [2:0] v176;
logic v177;
logic v178;
logic [63:0] boot_pc__linx_cpu_pyc__L25;
logic [63:0] boot_sp__linx_cpu_pyc__L26;
logic [2:0] state__stage__next;
logic [2:0] v179;
logic [2:0] state__stage;
logic [63:0] state__pc__next;
logic [63:0] v180;
logic [63:0] state__pc;
logic [1:0] state__br_kind__next;
logic [1:0] v181;
logic [1:0] state__br_kind;
logic [63:0] state__br_base_pc__next;
logic [63:0] v182;
logic [63:0] state__br_base_pc;
logic [63:0] state__br_off__next;
logic [63:0] v183;
logic [63:0] state__br_off;
logic state__commit_cond__next;
logic v184;
logic state__commit_cond;
logic [63:0] state__commit_tgt__next;
logic [63:0] v185;
logic [63:0] state__commit_tgt;
logic [63:0] state__cycles__next;
logic [63:0] v186;
logic [63:0] state__cycles;
logic state__halted__next;
logic v187;
logic state__halted;
logic [63:0] ifid__window__next;
logic [63:0] v188;
logic [63:0] ifid__window;
logic [5:0] idex__op__next;
logic [5:0] v189;
logic [5:0] idex__op;
logic [2:0] idex__len_bytes__next;
logic [2:0] v190;
logic [2:0] idex__len_bytes;
logic [5:0] idex__regdst__next;
logic [5:0] v191;
logic [5:0] idex__regdst;
logic [5:0] idex__srcl__next;
logic [5:0] v192;
logic [5:0] idex__srcl;
logic [5:0] idex__srcr__next;
logic [5:0] v193;
logic [5:0] idex__srcr;
logic [5:0] idex__srcp__next;
logic [5:0] v194;
logic [5:0] idex__srcp;
logic [63:0] idex__imm__next;
logic [63:0] v195;
logic [63:0] idex__imm;
logic [63:0] idex__srcl_val__next;
logic [63:0] v196;
logic [63:0] idex__srcl_val;
logic [63:0] idex__srcr_val__next;
logic [63:0] v197;
logic [63:0] idex__srcr_val;
logic [63:0] idex__srcp_val__next;
logic [63:0] v198;
logic [63:0] idex__srcp_val;
logic [5:0] exmem__op__next;
logic [5:0] v199;
logic [5:0] exmem__op;
logic [2:0] exmem__len_bytes__next;
logic [2:0] v200;
logic [2:0] exmem__len_bytes;
logic [5:0] exmem__regdst__next;
logic [5:0] v201;
logic [5:0] exmem__regdst;
logic [63:0] exmem__alu__next;
logic [63:0] v202;
logic [63:0] exmem__alu;
logic exmem__is_load__next;
logic v203;
logic exmem__is_load;
logic exmem__is_store__next;
logic v204;
logic exmem__is_store;
logic [2:0] exmem__size__next;
logic [2:0] v205;
logic [2:0] exmem__size;
logic [63:0] exmem__addr__next;
logic [63:0] v206;
logic [63:0] exmem__addr;
logic [63:0] exmem__wdata__next;
logic [63:0] v207;
logic [63:0] exmem__wdata;
logic [5:0] memwb__op__next;
logic [5:0] v208;
logic [5:0] memwb__op;
logic [2:0] memwb__len_bytes__next;
logic [2:0] v209;
logic [2:0] memwb__len_bytes;
logic [5:0] memwb__regdst__next;
logic [5:0] v210;
logic [5:0] memwb__regdst;
logic [63:0] memwb__value__next;
logic [63:0] v211;
logic [63:0] memwb__value;
logic [63:0] gpr__r0__next;
logic [63:0] v212;
logic [63:0] gpr__r1__next;
logic [63:0] v213;
logic [63:0] gpr__r1;
logic [63:0] gpr__r2__next;
logic [63:0] v214;
logic [63:0] gpr__r2;
logic [63:0] gpr__r3__next;
logic [63:0] v215;
logic [63:0] gpr__r3;
logic [63:0] gpr__r4__next;
logic [63:0] v216;
logic [63:0] gpr__r4;
logic [63:0] gpr__r5__next;
logic [63:0] v217;
logic [63:0] gpr__r5;
logic [63:0] gpr__r6__next;
logic [63:0] v218;
logic [63:0] gpr__r6;
logic [63:0] gpr__r7__next;
logic [63:0] v219;
logic [63:0] gpr__r7;
logic [63:0] gpr__r8__next;
logic [63:0] v220;
logic [63:0] gpr__r8;
logic [63:0] gpr__r9__next;
logic [63:0] v221;
logic [63:0] gpr__r9;
logic [63:0] gpr__r10__next;
logic [63:0] v222;
logic [63:0] gpr__r10;
logic [63:0] gpr__r11__next;
logic [63:0] v223;
logic [63:0] gpr__r11;
logic [63:0] gpr__r12__next;
logic [63:0] v224;
logic [63:0] gpr__r12;
logic [63:0] gpr__r13__next;
logic [63:0] v225;
logic [63:0] gpr__r13;
logic [63:0] gpr__r14__next;
logic [63:0] v226;
logic [63:0] gpr__r14;
logic [63:0] gpr__r15__next;
logic [63:0] v227;
logic [63:0] gpr__r15;
logic [63:0] gpr__r16__next;
logic [63:0] v228;
logic [63:0] gpr__r16;
logic [63:0] gpr__r17__next;
logic [63:0] v229;
logic [63:0] gpr__r17;
logic [63:0] gpr__r18__next;
logic [63:0] v230;
logic [63:0] gpr__r18;
logic [63:0] gpr__r19__next;
logic [63:0] v231;
logic [63:0] gpr__r19;
logic [63:0] gpr__r20__next;
logic [63:0] v232;
logic [63:0] gpr__r20;
logic [63:0] gpr__r21__next;
logic [63:0] v233;
logic [63:0] gpr__r21;
logic [63:0] gpr__r22__next;
logic [63:0] v234;
logic [63:0] gpr__r22;
logic [63:0] gpr__r23__next;
logic [63:0] v235;
logic [63:0] gpr__r23;
logic [63:0] t__r0__next;
logic [63:0] v236;
logic [63:0] t__r0;
logic [63:0] t__r1__next;
logic [63:0] v237;
logic [63:0] t__r1;
logic [63:0] t__r2__next;
logic [63:0] v238;
logic [63:0] t__r2;
logic [63:0] t__r3__next;
logic [63:0] v239;
logic [63:0] t__r3;
logic [63:0] u__r0__next;
logic [63:0] v240;
logic [63:0] u__r0;
logic [63:0] u__r1__next;
logic [63:0] v241;
logic [63:0] u__r1;
logic [63:0] u__r2__next;
logic [63:0] v242;
logic [63:0] u__r2;
logic [63:0] u__r3__next;
logic [63:0] v243;
logic [63:0] u__r3;
logic v244;
logic stage_is_if__linx_cpu_pyc__L94;
logic v245;
logic stage_is_id__linx_cpu_pyc__L95;
logic v246;
logic stage_is_ex__linx_cpu_pyc__L96;
logic v247;
logic stage_is_mem__linx_cpu_pyc__L97;
logic v248;
logic stage_is_wb__linx_cpu_pyc__L98;
logic v249;
logic v250;
logic v251;
logic v252;
logic v253;
logic v254;
logic v255;
logic halt_set__linx_cpu_pyc__L100;
logic v256;
logic stop__linx_cpu_pyc__L101;
logic v257;
logic active__linx_cpu_pyc__L102;
logic v258;
logic do_if__linx_cpu_pyc__L104;
logic v259;
logic do_id__linx_cpu_pyc__L105;
logic v260;
logic do_ex__linx_cpu_pyc__L106;
logic v261;
logic do_mem__linx_cpu_pyc__L107;
logic v262;
logic do_wb__linx_cpu_pyc__L108;
logic v263;
logic [63:0] v264;
logic [63:0] v265;
logic [63:0] v266;
logic [63:0] mem_raddr__linx_cpu_pyc__L111;
logic v267;
logic mem_wvalid__linx_cpu_pyc__L112;
logic [63:0] mem_waddr__linx_cpu_pyc__L113;
logic [63:0] mem_wdata__linx_cpu_pyc__L114;
logic v268;
logic [7:0] v269;
logic [7:0] v270;
logic [7:0] mem_wstrb__linx_cpu_pyc__L115;
logic v271;
logic [7:0] v272;
logic [7:0] v273;
logic [7:0] mem_wstrb__linx_cpu_pyc__L116;
logic [63:0] v274;
logic [63:0] mem_rdata__linx_cpu_pyc__L118;
logic [63:0] v275;
logic [15:0] v276;
logic [31:0] v277;
logic [47:0] v278;
logic [3:0] v279;
logic v280;
logic v281;
logic v282;
logic v283;
logic v284;
logic v285;
logic [4:0] v286;
logic [5:0] v287;
logic [4:0] v288;
logic [5:0] v289;
logic [4:0] v290;
logic [5:0] v291;
logic [4:0] v292;
logic [5:0] v293;
logic [11:0] v294;
logic [63:0] v295;
logic [63:0] v296;
logic [19:0] v297;
logic [63:0] v298;
logic [6:0] v299;
logic [11:0] v300;
logic [11:0] v301;
logic [11:0] v302;
logic [11:0] v303;
logic [63:0] v304;
logic [16:0] v305;
logic [63:0] v306;
logic [15:0] v307;
logic [31:0] v308;
logic [11:0] v309;
logic [19:0] v310;
logic [31:0] v311;
logic [31:0] v312;
logic [31:0] v313;
logic [31:0] v314;
logic [63:0] v315;
logic [4:0] v316;
logic [5:0] v317;
logic [4:0] v318;
logic [5:0] v319;
logic [4:0] v320;
logic [5:0] v321;
logic [63:0] v322;
logic [63:0] v323;
logic [11:0] v324;
logic [63:0] v325;
logic [63:0] v326;
logic [2:0] v327;
logic [63:0] v328;
logic [15:0] v329;
logic v330;
logic v331;
logic [5:0] v332;
logic [2:0] v333;
logic [5:0] v334;
logic [5:0] v335;
logic [63:0] v336;
logic [15:0] v337;
logic v338;
logic v339;
logic [63:0] v340;
logic [5:0] v341;
logic [2:0] v342;
logic [5:0] v343;
logic [5:0] v344;
logic [63:0] v345;
logic v346;
logic v347;
logic [5:0] v348;
logic [2:0] v349;
logic [5:0] v350;
logic [5:0] v351;
logic [5:0] v352;
logic [5:0] v353;
logic [63:0] v354;
logic v355;
logic v356;
logic [5:0] v357;
logic [2:0] v358;
logic [5:0] v359;
logic [5:0] v360;
logic [5:0] v361;
logic [5:0] v362;
logic [63:0] v363;
logic v364;
logic v365;
logic [5:0] v366;
logic [2:0] v367;
logic [5:0] v368;
logic [5:0] v369;
logic [5:0] v370;
logic [5:0] v371;
logic [63:0] v372;
logic v373;
logic v374;
logic [5:0] v375;
logic [2:0] v376;
logic [5:0] v377;
logic [5:0] v378;
logic [5:0] v379;
logic [5:0] v380;
logic [63:0] v381;
logic v382;
logic v383;
logic [5:0] v384;
logic [2:0] v385;
logic [5:0] v386;
logic [5:0] v387;
logic [5:0] v388;
logic [5:0] v389;
logic [63:0] v390;
logic [15:0] v391;
logic v392;
logic v393;
logic [63:0] v394;
logic [5:0] v395;
logic [2:0] v396;
logic [5:0] v397;
logic [5:0] v398;
logic [5:0] v399;
logic [5:0] v400;
logic [63:0] v401;
logic [15:0] v402;
logic v403;
logic v404;
logic [5:0] v405;
logic [2:0] v406;
logic [5:0] v407;
logic [5:0] v408;
logic [5:0] v409;
logic [5:0] v410;
logic [63:0] v411;
logic [15:0] v412;
logic v413;
logic v414;
logic [5:0] v415;
logic [2:0] v416;
logic [5:0] v417;
logic [5:0] v418;
logic [5:0] v419;
logic [5:0] v420;
logic [63:0] v421;
logic [31:0] v422;
logic v423;
logic v424;
logic [5:0] v425;
logic [2:0] v426;
logic [5:0] v427;
logic [5:0] v428;
logic [5:0] v429;
logic [5:0] v430;
logic [63:0] v431;
logic v432;
logic v433;
logic [5:0] v434;
logic [2:0] v435;
logic [5:0] v436;
logic [5:0] v437;
logic [5:0] v438;
logic [5:0] v439;
logic [63:0] v440;
logic v441;
logic v442;
logic [5:0] v443;
logic [2:0] v444;
logic [5:0] v445;
logic [5:0] v446;
logic [5:0] v447;
logic [5:0] v448;
logic [63:0] v449;
logic v450;
logic v451;
logic [5:0] v452;
logic [2:0] v453;
logic [5:0] v454;
logic [5:0] v455;
logic [5:0] v456;
logic [5:0] v457;
logic [63:0] v458;
logic v459;
logic v460;
logic [5:0] v461;
logic [2:0] v462;
logic [5:0] v463;
logic [5:0] v464;
logic [5:0] v465;
logic [5:0] v466;
logic [63:0] v467;
logic v468;
logic v469;
logic [5:0] v470;
logic [2:0] v471;
logic [5:0] v472;
logic [5:0] v473;
logic [5:0] v474;
logic [5:0] v475;
logic [63:0] v476;
logic v477;
logic v478;
logic [5:0] v479;
logic [2:0] v480;
logic [5:0] v481;
logic [5:0] v482;
logic [5:0] v483;
logic [5:0] v484;
logic [63:0] v485;
logic v486;
logic v487;
logic [5:0] v488;
logic [2:0] v489;
logic [5:0] v490;
logic [5:0] v491;
logic [5:0] v492;
logic [5:0] v493;
logic [63:0] v494;
logic v495;
logic v496;
logic [5:0] v497;
logic [2:0] v498;
logic [5:0] v499;
logic [5:0] v500;
logic [5:0] v501;
logic [5:0] v502;
logic [63:0] v503;
logic v504;
logic v505;
logic [5:0] v506;
logic [2:0] v507;
logic [5:0] v508;
logic [5:0] v509;
logic [5:0] v510;
logic [5:0] v511;
logic [63:0] v512;
logic v513;
logic v514;
logic [5:0] v515;
logic [2:0] v516;
logic [5:0] v517;
logic [5:0] v518;
logic [5:0] v519;
logic [5:0] v520;
logic [63:0] v521;
logic [31:0] v522;
logic v523;
logic v524;
logic [5:0] v525;
logic [2:0] v526;
logic [5:0] v527;
logic [5:0] v528;
logic [5:0] v529;
logic [5:0] v530;
logic [63:0] v531;
logic [31:0] v532;
logic v533;
logic v534;
logic [5:0] v535;
logic [2:0] v536;
logic [5:0] v537;
logic [5:0] v538;
logic [5:0] v539;
logic [5:0] v540;
logic [63:0] v541;
logic [31:0] v542;
logic v543;
logic v544;
logic [63:0] v545;
logic [5:0] v546;
logic [2:0] v547;
logic [5:0] v548;
logic [5:0] v549;
logic [5:0] v550;
logic [5:0] v551;
logic [63:0] v552;
logic [31:0] v553;
logic v554;
logic v555;
logic [63:0] v556;
logic [5:0] v557;
logic [2:0] v558;
logic [5:0] v559;
logic [5:0] v560;
logic [5:0] v561;
logic [5:0] v562;
logic [63:0] v563;
logic [47:0] v564;
logic v565;
logic v566;
logic [5:0] v567;
logic [2:0] v568;
logic [5:0] v569;
logic [5:0] v570;
logic [5:0] v571;
logic [5:0] v572;
logic [63:0] v573;
logic [5:0] v574;
logic [2:0] v575;
logic [5:0] v576;
logic [5:0] v577;
logic [5:0] v578;
logic [5:0] v579;
logic [63:0] v580;
logic [5:0] v581;
logic [2:0] v582;
logic [5:0] v583;
logic [5:0] v584;
logic [5:0] v585;
logic [5:0] v586;
logic [63:0] v587;
logic v588;
logic [63:0] v589;
logic v590;
logic [63:0] v591;
logic v592;
logic [63:0] v593;
logic v594;
logic [63:0] v595;
logic v596;
logic [63:0] v597;
logic v598;
logic [63:0] v599;
logic v600;
logic [63:0] v601;
logic v602;
logic [63:0] v603;
logic v604;
logic [63:0] v605;
logic v606;
logic [63:0] v607;
logic v608;
logic [63:0] v609;
logic v610;
logic [63:0] v611;
logic v612;
logic [63:0] v613;
logic v614;
logic [63:0] v615;
logic v616;
logic [63:0] v617;
logic v618;
logic [63:0] v619;
logic v620;
logic [63:0] v621;
logic v622;
logic [63:0] v623;
logic v624;
logic [63:0] v625;
logic v626;
logic [63:0] v627;
logic v628;
logic [63:0] v629;
logic v630;
logic [63:0] v631;
logic v632;
logic [63:0] v633;
logic v634;
logic [63:0] v635;
logic v636;
logic [63:0] v637;
logic v638;
logic [63:0] v639;
logic v640;
logic [63:0] v641;
logic v642;
logic [63:0] v643;
logic v644;
logic [63:0] v645;
logic v646;
logic [63:0] v647;
logic v648;
logic [63:0] v649;
logic v650;
logic [63:0] v651;
logic v652;
logic [63:0] v653;
logic v654;
logic [63:0] v655;
logic v656;
logic [63:0] v657;
logic v658;
logic [63:0] v659;
logic v660;
logic [63:0] v661;
logic v662;
logic [63:0] v663;
logic v664;
logic [63:0] v665;
logic v666;
logic [63:0] v667;
logic v668;
logic [63:0] v669;
logic v670;
logic [63:0] v671;
logic v672;
logic [63:0] v673;
logic v674;
logic [63:0] v675;
logic v676;
logic [63:0] v677;
logic v678;
logic [63:0] v679;
logic v680;
logic [63:0] v681;
logic v682;
logic [63:0] v683;
logic v684;
logic [63:0] v685;
logic v686;
logic [63:0] v687;
logic v688;
logic [63:0] v689;
logic v690;
logic [63:0] v691;
logic v692;
logic [63:0] v693;
logic v694;
logic [63:0] v695;
logic v696;
logic [63:0] v697;
logic v698;
logic [63:0] v699;
logic v700;
logic [63:0] v701;
logic v702;
logic [63:0] v703;
logic v704;
logic [63:0] v705;
logic v706;
logic [63:0] v707;
logic v708;
logic [63:0] v709;
logic v710;
logic [63:0] v711;
logic v712;
logic [63:0] v713;
logic v714;
logic [63:0] v715;
logic v716;
logic [63:0] v717;
logic v718;
logic [63:0] v719;
logic v720;
logic [63:0] v721;
logic v722;
logic [63:0] v723;
logic v724;
logic [63:0] v725;
logic v726;
logic [63:0] v727;
logic v728;
logic [63:0] v729;
logic v730;
logic [63:0] v731;
logic v732;
logic [63:0] v733;
logic v734;
logic [63:0] v735;
logic v736;
logic [63:0] v737;
logic v738;
logic [63:0] v739;
logic v740;
logic [63:0] v741;
logic v742;
logic [63:0] v743;
logic v744;
logic [63:0] v745;
logic v746;
logic [63:0] v747;
logic v748;
logic [63:0] v749;
logic v750;
logic [63:0] v751;
logic v752;
logic [63:0] v753;
logic v754;
logic [63:0] v755;
logic v756;
logic [63:0] v757;
logic v758;
logic [63:0] v759;
logic v760;
logic [63:0] v761;
logic v762;
logic [63:0] v763;
logic v764;
logic [63:0] v765;
logic v766;
logic [63:0] v767;
logic v768;
logic [63:0] v769;
logic v770;
logic [63:0] v771;
logic v772;
logic [63:0] v773;
logic v774;
logic [63:0] v775;
logic v776;
logic [63:0] v777;
logic v778;
logic [63:0] v779;
logic [63:0] v780;
logic [63:0] v781;
logic [63:0] v782;
logic [63:0] v783;
logic [63:0] v784;
logic [63:0] v785;
logic v786;
logic v787;
logic v788;
logic v789;
logic v790;
logic v791;
logic v792;
logic v793;
logic v794;
logic v795;
logic v796;
logic v797;
logic v798;
logic v799;
logic v800;
logic v801;
logic v802;
logic v803;
logic v804;
logic v805;
logic v806;
logic v807;
logic v808;
logic v809;
logic [63:0] v810;
logic v811;
logic v812;
logic [63:0] v813;
logic v814;
logic [2:0] v815;
logic [63:0] v816;
logic [63:0] v817;
logic v818;
logic [2:0] v819;
logic [63:0] v820;
logic [63:0] v821;
logic v822;
logic [2:0] v823;
logic [63:0] v824;
logic [63:0] v825;
logic [63:0] v826;
logic v827;
logic [2:0] v828;
logic [63:0] v829;
logic v830;
logic [63:0] v831;
logic [63:0] v832;
logic v833;
logic [2:0] v834;
logic [63:0] v835;
logic [63:0] v836;
logic v837;
logic [2:0] v838;
logic [63:0] v839;
logic [63:0] v840;
logic [63:0] v841;
logic [63:0] v842;
logic v843;
logic [2:0] v844;
logic [63:0] v845;
logic [63:0] v846;
logic [63:0] v847;
logic v848;
logic [2:0] v849;
logic [63:0] v850;
logic [63:0] v851;
logic [63:0] v852;
logic [63:0] v853;
logic [63:0] v854;
logic v855;
logic [2:0] v856;
logic [63:0] v857;
logic [31:0] v858;
logic [31:0] v859;
logic [31:0] v860;
logic [63:0] v861;
logic [63:0] v862;
logic v863;
logic [2:0] v864;
logic [63:0] v865;
logic [31:0] v866;
logic [31:0] v867;
logic [63:0] v868;
logic [31:0] v869;
logic [63:0] v870;
logic [31:0] v871;
logic [63:0] v872;
logic [31:0] v873;
logic [63:0] v874;
logic [63:0] v875;
logic v876;
logic [2:0] v877;
logic [63:0] v878;
logic [63:0] v879;
logic v880;
logic [2:0] v881;
logic [63:0] v882;
logic [63:0] v883;
logic v884;
logic [2:0] v885;
logic [63:0] v886;
logic [63:0] v887;
logic v888;
logic [2:0] v889;
logic [63:0] v890;
logic [63:0] v891;
logic v892;
logic [2:0] v893;
logic [63:0] v894;
logic [63:0] v895;
logic v896;
logic [2:0] v897;
logic [63:0] v898;
logic v899;
logic v900;
logic [63:0] v901;
logic [63:0] v902;
logic v903;
logic [2:0] v904;
logic [63:0] v905;
logic v906;
logic [63:0] v907;
logic [63:0] v908;
logic v909;
logic v910;
logic [2:0] v911;
logic [63:0] v912;
logic [63:0] v913;
logic [63:0] v914;
logic [63:0] v915;
logic [63:0] v916;
logic v917;
logic [63:0] v918;
logic v919;
logic v920;
logic [2:0] v921;
logic [63:0] v922;
logic [63:0] v923;
logic [63:0] v924;
logic [63:0] v925;
logic [63:0] v926;
logic v927;
logic v928;
logic [2:0] v929;
logic [63:0] v930;
logic [63:0] v931;
logic [5:0] v932;
logic [63:0] v933;
logic v934;
logic v935;
logic [2:0] v936;
logic [63:0] v937;
logic [63:0] v938;
logic [5:0] v939;
logic [2:0] v940;
logic [5:0] v941;
logic [63:0] v942;
logic v943;
logic v944;
logic [2:0] v945;
logic [63:0] v946;
logic [63:0] v947;
logic [31:0] v948;
logic [63:0] v949;
logic [63:0] v950;
logic [63:0] v951;
logic [5:0] v952;
logic [63:0] v953;
logic [5:0] v954;
logic [2:0] v955;
logic [5:0] v956;
logic [63:0] v957;
logic v958;
logic v959;
logic v960;
logic v961;
logic v962;
logic v963;
logic v964;
logic v965;
logic v966;
logic v967;
logic v968;
logic [63:0] v969;
logic [63:0] v970;
logic v971;
logic v972;
logic v973;
logic [63:0] v974;
logic [63:0] v975;
logic [63:0] v976;
logic [63:0] v977;
logic [63:0] v978;
logic v979;
logic v980;
logic v981;
logic v982;
logic v983;
logic v984;
logic v985;
logic [63:0] v986;
logic [2:0] v987;
logic [2:0] v988;
logic [2:0] v989;
logic [2:0] v990;
logic [2:0] v991;
logic [2:0] v992;
logic [2:0] v993;
logic [63:0] v994;
logic v995;
logic v996;
logic v997;
logic v998;
logic [63:0] v999;
logic v1000;
logic v1001;
logic v1002;
logic v1003;
logic [63:0] v1004;
logic v1005;
logic v1006;
logic [63:0] v1007;
logic v1008;
logic [1:0] v1009;
logic [63:0] v1010;
logic [63:0] v1011;
logic v1012;
logic v1013;
logic v1014;
logic v1015;
logic [1:0] v1016;
logic [63:0] v1017;
logic [63:0] v1018;
logic v1019;
logic [1:0] v1020;
logic [63:0] v1021;
logic [63:0] v1022;
logic [2:0] v1023;
logic v1024;
logic [1:0] v1025;
logic v1026;
logic [1:0] v1027;
logic [63:0] v1028;
logic [63:0] v1029;
logic v1030;
logic [1:0] v1031;
logic [63:0] v1032;
logic [63:0] v1033;
logic v1034;
logic [1:0] v1035;
logic [63:0] v1036;
logic [63:0] v1037;
logic v1038;
logic v1039;
logic v1040;
logic v1041;
logic v1042;
logic v1043;
logic v1044;
logic v1045;
logic v1046;
logic v1047;
logic v1048;
logic v1049;
logic v1050;
logic v1051;
logic v1052;
logic v1053;
logic v1054;
logic v1055;
logic v1056;
logic v1057;
logic [63:0] v1058;
logic [63:0] v1059;
logic v1060;
logic v1061;
logic [63:0] v1062;
logic [63:0] v1063;
logic v1064;
logic v1065;
logic [63:0] v1066;
logic [63:0] v1067;
logic v1068;
logic v1069;
logic [63:0] v1070;
logic [63:0] v1071;
logic v1072;
logic v1073;
logic [63:0] v1074;
logic [63:0] v1075;
logic v1076;
logic v1077;
logic [63:0] v1078;
logic [63:0] v1079;
logic v1080;
logic v1081;
logic [63:0] v1082;
logic [63:0] v1083;
logic v1084;
logic v1085;
logic [63:0] v1086;
logic [63:0] v1087;
logic v1088;
logic v1089;
logic [63:0] v1090;
logic [63:0] v1091;
logic v1092;
logic v1093;
logic [63:0] v1094;
logic [63:0] v1095;
logic v1096;
logic v1097;
logic [63:0] v1098;
logic [63:0] v1099;
logic v1100;
logic v1101;
logic [63:0] v1102;
logic [63:0] v1103;
logic v1104;
logic v1105;
logic [63:0] v1106;
logic [63:0] v1107;
logic v1108;
logic v1109;
logic [63:0] v1110;
logic [63:0] v1111;
logic v1112;
logic v1113;
logic [63:0] v1114;
logic [63:0] v1115;
logic v1116;
logic v1117;
logic [63:0] v1118;
logic [63:0] v1119;
logic v1120;
logic v1121;
logic [63:0] v1122;
logic [63:0] v1123;
logic v1124;
logic v1125;
logic [63:0] v1126;
logic [63:0] v1127;
logic v1128;
logic v1129;
logic [63:0] v1130;
logic [63:0] v1131;
logic v1132;
logic v1133;
logic [63:0] v1134;
logic [63:0] v1135;
logic v1136;
logic v1137;
logic [63:0] v1138;
logic [63:0] v1139;
logic v1140;
logic v1141;
logic [63:0] v1142;
logic [63:0] v1143;
logic v1144;
logic v1145;
logic [63:0] v1146;
logic [63:0] v1147;
logic [63:0] v1148;
logic [63:0] v1149;
logic [63:0] v1150;
logic [63:0] v1151;
logic [63:0] v1152;
logic [63:0] v1153;
logic [63:0] v1154;
logic [63:0] v1155;
logic [63:0] v1156;
logic [63:0] v1157;
logic [63:0] v1158;
logic [63:0] v1159;
logic [63:0] v1160;
logic [63:0] v1161;
logic [63:0] v1162;
logic [63:0] v1163;
logic [63:0] v1164;
logic [63:0] v1165;
logic [63:0] v1166;
logic [63:0] v1167;
logic [63:0] v1168;
logic [63:0] v1169;
logic [63:0] v1170;
logic [63:0] v1171;

assign v1 = 3'd7;
assign v2 = 2'd3;
assign v3 = 2'd2;
assign v4 = 2'd1;
assign v5 = 64'd18446744073709547520;
assign v6 = 6'd31;
assign v7 = 6'd30;
assign v8 = 6'd29;
assign v9 = 6'd28;
assign v10 = 6'd27;
assign v11 = 3'd6;
assign v12 = 6'd17;
assign v13 = 48'd1507342;
assign v14 = 48'd8323087;
assign v15 = 6'd20;
assign v16 = 32'd16385;
assign v17 = 32'd32767;
assign v18 = 6'd25;
assign v19 = 32'd7;
assign v20 = 32'd127;
assign v21 = 32'd1052715;
assign v22 = 32'd4043309055;
assign v23 = 6'd15;
assign v24 = 32'd69;
assign v25 = 32'd4160778367;
assign v26 = 6'd6;
assign v27 = 32'd4117;
assign v28 = 6'd7;
assign v29 = 32'd21;
assign v30 = 6'd8;
assign v31 = 32'd53;
assign v32 = 6'd9;
assign v33 = 32'd8217;
assign v34 = 6'd26;
assign v35 = 32'd12377;
assign v36 = 32'd8281;
assign v37 = 6'd16;
assign v38 = 32'd119;
assign v39 = 6'd11;
assign v40 = 32'd37;
assign v41 = 6'd12;
assign v42 = 32'd12325;
assign v43 = 6'd13;
assign v44 = 32'd8229;
assign v45 = 6'd14;
assign v46 = 32'd16421;
assign v47 = 32'd28799;
assign v48 = 6'd2;
assign v49 = 16'd65535;
assign v50 = 6'd1;
assign v51 = 16'd0;
assign v52 = 16'd51199;
assign v53 = 6'd19;
assign v54 = 16'd4;
assign v55 = 16'd15;
assign v56 = 16'd28;
assign v57 = 6'd23;
assign v58 = 16'd38;
assign v59 = 6'd3;
assign v60 = 16'd6;
assign v61 = 6'd4;
assign v62 = 16'd10;
assign v63 = 6'd5;
assign v64 = 6'd24;
assign v65 = 16'd42;
assign v66 = 6'd22;
assign v67 = 6'd10;
assign v68 = 16'd20502;
assign v69 = 16'd63551;
assign v70 = 6'd21;
assign v71 = 16'd22;
assign v72 = 16'd63;
assign v73 = 4'd14;
assign v74 = 8'd15;
assign v75 = 8'd255;
assign v76 = 6'd18;
assign v77 = 3'd4;
assign v78 = 3'd3;
assign v79 = 3'd2;
assign v80 = 3'd1;
assign v81 = 6'd63;
assign v82 = 2'd0;
assign v83 = 64'd1;
assign v84 = 64'd0;
assign v85 = 8'd0;
assign v86 = 6'd0;
assign v87 = 3'd0;
assign v88 = 1'd0;
assign v89 = 1'd1;
assign v90 = v1;
assign v91 = v2;
assign v92 = v3;
assign v93 = v4;
assign v94 = v5;
assign v95 = v6;
assign v96 = v7;
assign v97 = v8;
assign v98 = v9;
assign v99 = v10;
assign v100 = v11;
assign v101 = v12;
assign v102 = v13;
assign v103 = v14;
assign v104 = v15;
assign v105 = v16;
assign v106 = v17;
assign v107 = v18;
assign v108 = v19;
assign v109 = v20;
assign v110 = v21;
assign v111 = v22;
assign v112 = v23;
assign v113 = v24;
assign v114 = v25;
assign v115 = v26;
assign v116 = v27;
assign v117 = v28;
assign v118 = v29;
assign v119 = v30;
assign v120 = v31;
assign v121 = v32;
assign v122 = v33;
assign v123 = v34;
assign v124 = v35;
assign v125 = v36;
assign v126 = v37;
assign v127 = v38;
assign v128 = v39;
assign v129 = v40;
assign v130 = v41;
assign v131 = v42;
assign v132 = v43;
assign v133 = v44;
assign v134 = v45;
assign v135 = v46;
assign v136 = v47;
assign v137 = v48;
assign v138 = v49;
assign v139 = v50;
assign v140 = v51;
assign v141 = v52;
assign v142 = v53;
assign v143 = v54;
assign v144 = v55;
assign v145 = v56;
assign v146 = v57;
assign v147 = v58;
assign v148 = v59;
assign v149 = v60;
assign v150 = v61;
assign v151 = v62;
assign v152 = v63;
assign v153 = v64;
assign v154 = v65;
assign v155 = v66;
assign v156 = v67;
assign v157 = v68;
assign v158 = v69;
assign v159 = v70;
assign v160 = v71;
assign v161 = v72;
assign v162 = v73;
assign v163 = v74;
assign v164 = v75;
assign v165 = v76;
assign v166 = v77;
assign v167 = v78;
assign v168 = v79;
assign v169 = v80;
assign v170 = v81;
assign v171 = v82;
assign v172 = v83;
assign v173 = v84;
assign v174 = v85;
assign v175 = v86;
assign v176 = v87;
assign v177 = v88;
assign v178 = v89;
assign boot_pc__linx_cpu_pyc__L25 = boot_pc;
assign boot_sp__linx_cpu_pyc__L26 = boot_sp;
pyc_reg #(.WIDTH(3)) v179_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(state__stage__next),
  .init(v176),
  .q(v179)
);
assign state__stage = v179;
pyc_reg #(.WIDTH(64)) v180_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(state__pc__next),
  .init(boot_pc__linx_cpu_pyc__L25),
  .q(v180)
);
assign state__pc = v180;
pyc_reg #(.WIDTH(2)) v181_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(state__br_kind__next),
  .init(v171),
  .q(v181)
);
assign state__br_kind = v181;
pyc_reg #(.WIDTH(64)) v182_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(state__br_base_pc__next),
  .init(boot_pc__linx_cpu_pyc__L25),
  .q(v182)
);
assign state__br_base_pc = v182;
pyc_reg #(.WIDTH(64)) v183_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(state__br_off__next),
  .init(v173),
  .q(v183)
);
assign state__br_off = v183;
pyc_reg #(.WIDTH(1)) v184_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(state__commit_cond__next),
  .init(v177),
  .q(v184)
);
assign state__commit_cond = v184;
pyc_reg #(.WIDTH(64)) v185_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(state__commit_tgt__next),
  .init(v173),
  .q(v185)
);
assign state__commit_tgt = v185;
pyc_reg #(.WIDTH(64)) v186_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(state__cycles__next),
  .init(v173),
  .q(v186)
);
assign state__cycles = v186;
pyc_reg #(.WIDTH(1)) v187_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(state__halted__next),
  .init(v177),
  .q(v187)
);
assign state__halted = v187;
pyc_reg #(.WIDTH(64)) v188_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(ifid__window__next),
  .init(v173),
  .q(v188)
);
assign ifid__window = v188;
pyc_reg #(.WIDTH(6)) v189_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(idex__op__next),
  .init(v175),
  .q(v189)
);
assign idex__op = v189;
pyc_reg #(.WIDTH(3)) v190_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(idex__len_bytes__next),
  .init(v176),
  .q(v190)
);
assign idex__len_bytes = v190;
pyc_reg #(.WIDTH(6)) v191_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(idex__regdst__next),
  .init(v170),
  .q(v191)
);
assign idex__regdst = v191;
pyc_reg #(.WIDTH(6)) v192_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(idex__srcl__next),
  .init(v170),
  .q(v192)
);
assign idex__srcl = v192;
pyc_reg #(.WIDTH(6)) v193_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(idex__srcr__next),
  .init(v170),
  .q(v193)
);
assign idex__srcr = v193;
pyc_reg #(.WIDTH(6)) v194_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(idex__srcp__next),
  .init(v170),
  .q(v194)
);
assign idex__srcp = v194;
pyc_reg #(.WIDTH(64)) v195_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(idex__imm__next),
  .init(v173),
  .q(v195)
);
assign idex__imm = v195;
pyc_reg #(.WIDTH(64)) v196_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(idex__srcl_val__next),
  .init(v173),
  .q(v196)
);
assign idex__srcl_val = v196;
pyc_reg #(.WIDTH(64)) v197_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(idex__srcr_val__next),
  .init(v173),
  .q(v197)
);
assign idex__srcr_val = v197;
pyc_reg #(.WIDTH(64)) v198_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(idex__srcp_val__next),
  .init(v173),
  .q(v198)
);
assign idex__srcp_val = v198;
pyc_reg #(.WIDTH(6)) v199_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(exmem__op__next),
  .init(v175),
  .q(v199)
);
assign exmem__op = v199;
pyc_reg #(.WIDTH(3)) v200_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(exmem__len_bytes__next),
  .init(v176),
  .q(v200)
);
assign exmem__len_bytes = v200;
pyc_reg #(.WIDTH(6)) v201_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(exmem__regdst__next),
  .init(v170),
  .q(v201)
);
assign exmem__regdst = v201;
pyc_reg #(.WIDTH(64)) v202_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(exmem__alu__next),
  .init(v173),
  .q(v202)
);
assign exmem__alu = v202;
pyc_reg #(.WIDTH(1)) v203_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(exmem__is_load__next),
  .init(v177),
  .q(v203)
);
assign exmem__is_load = v203;
pyc_reg #(.WIDTH(1)) v204_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(exmem__is_store__next),
  .init(v177),
  .q(v204)
);
assign exmem__is_store = v204;
pyc_reg #(.WIDTH(3)) v205_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(exmem__size__next),
  .init(v176),
  .q(v205)
);
assign exmem__size = v205;
pyc_reg #(.WIDTH(64)) v206_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(exmem__addr__next),
  .init(v173),
  .q(v206)
);
assign exmem__addr = v206;
pyc_reg #(.WIDTH(64)) v207_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(exmem__wdata__next),
  .init(v173),
  .q(v207)
);
assign exmem__wdata = v207;
pyc_reg #(.WIDTH(6)) v208_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(memwb__op__next),
  .init(v175),
  .q(v208)
);
assign memwb__op = v208;
pyc_reg #(.WIDTH(3)) v209_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(memwb__len_bytes__next),
  .init(v176),
  .q(v209)
);
assign memwb__len_bytes = v209;
pyc_reg #(.WIDTH(6)) v210_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(memwb__regdst__next),
  .init(v170),
  .q(v210)
);
assign memwb__regdst = v210;
pyc_reg #(.WIDTH(64)) v211_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(memwb__value__next),
  .init(v173),
  .q(v211)
);
assign memwb__value = v211;
pyc_reg #(.WIDTH(64)) v212_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r0__next),
  .init(v173),
  .q(v212)
);
pyc_reg #(.WIDTH(64)) v213_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r1__next),
  .init(boot_sp__linx_cpu_pyc__L26),
  .q(v213)
);
assign gpr__r1 = v213;
pyc_reg #(.WIDTH(64)) v214_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r2__next),
  .init(v173),
  .q(v214)
);
assign gpr__r2 = v214;
pyc_reg #(.WIDTH(64)) v215_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r3__next),
  .init(v173),
  .q(v215)
);
assign gpr__r3 = v215;
pyc_reg #(.WIDTH(64)) v216_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r4__next),
  .init(v173),
  .q(v216)
);
assign gpr__r4 = v216;
pyc_reg #(.WIDTH(64)) v217_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r5__next),
  .init(v173),
  .q(v217)
);
assign gpr__r5 = v217;
pyc_reg #(.WIDTH(64)) v218_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r6__next),
  .init(v173),
  .q(v218)
);
assign gpr__r6 = v218;
pyc_reg #(.WIDTH(64)) v219_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r7__next),
  .init(v173),
  .q(v219)
);
assign gpr__r7 = v219;
pyc_reg #(.WIDTH(64)) v220_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r8__next),
  .init(v173),
  .q(v220)
);
assign gpr__r8 = v220;
pyc_reg #(.WIDTH(64)) v221_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r9__next),
  .init(v173),
  .q(v221)
);
assign gpr__r9 = v221;
pyc_reg #(.WIDTH(64)) v222_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r10__next),
  .init(v173),
  .q(v222)
);
assign gpr__r10 = v222;
pyc_reg #(.WIDTH(64)) v223_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r11__next),
  .init(v173),
  .q(v223)
);
assign gpr__r11 = v223;
pyc_reg #(.WIDTH(64)) v224_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r12__next),
  .init(v173),
  .q(v224)
);
assign gpr__r12 = v224;
pyc_reg #(.WIDTH(64)) v225_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r13__next),
  .init(v173),
  .q(v225)
);
assign gpr__r13 = v225;
pyc_reg #(.WIDTH(64)) v226_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r14__next),
  .init(v173),
  .q(v226)
);
assign gpr__r14 = v226;
pyc_reg #(.WIDTH(64)) v227_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r15__next),
  .init(v173),
  .q(v227)
);
assign gpr__r15 = v227;
pyc_reg #(.WIDTH(64)) v228_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r16__next),
  .init(v173),
  .q(v228)
);
assign gpr__r16 = v228;
pyc_reg #(.WIDTH(64)) v229_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r17__next),
  .init(v173),
  .q(v229)
);
assign gpr__r17 = v229;
pyc_reg #(.WIDTH(64)) v230_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r18__next),
  .init(v173),
  .q(v230)
);
assign gpr__r18 = v230;
pyc_reg #(.WIDTH(64)) v231_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r19__next),
  .init(v173),
  .q(v231)
);
assign gpr__r19 = v231;
pyc_reg #(.WIDTH(64)) v232_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r20__next),
  .init(v173),
  .q(v232)
);
assign gpr__r20 = v232;
pyc_reg #(.WIDTH(64)) v233_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r21__next),
  .init(v173),
  .q(v233)
);
assign gpr__r21 = v233;
pyc_reg #(.WIDTH(64)) v234_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r22__next),
  .init(v173),
  .q(v234)
);
assign gpr__r22 = v234;
pyc_reg #(.WIDTH(64)) v235_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(gpr__r23__next),
  .init(v173),
  .q(v235)
);
assign gpr__r23 = v235;
pyc_reg #(.WIDTH(64)) v236_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(t__r0__next),
  .init(v173),
  .q(v236)
);
assign t__r0 = v236;
pyc_reg #(.WIDTH(64)) v237_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(t__r1__next),
  .init(v173),
  .q(v237)
);
assign t__r1 = v237;
pyc_reg #(.WIDTH(64)) v238_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(t__r2__next),
  .init(v173),
  .q(v238)
);
assign t__r2 = v238;
pyc_reg #(.WIDTH(64)) v239_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(t__r3__next),
  .init(v173),
  .q(v239)
);
assign t__r3 = v239;
pyc_reg #(.WIDTH(64)) v240_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(u__r0__next),
  .init(v173),
  .q(v240)
);
assign u__r0 = v240;
pyc_reg #(.WIDTH(64)) v241_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(u__r1__next),
  .init(v173),
  .q(v241)
);
assign u__r1 = v241;
pyc_reg #(.WIDTH(64)) v242_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(u__r2__next),
  .init(v173),
  .q(v242)
);
assign u__r2 = v242;
pyc_reg #(.WIDTH(64)) v243_inst (
  .clk(clk),
  .rst(rst),
  .en(v178),
  .d(u__r3__next),
  .init(v173),
  .q(v243)
);
assign u__r3 = v243;
assign v244 = (state__stage == v176);
assign stage_is_if__linx_cpu_pyc__L94 = v244;
assign v245 = (state__stage == v169);
assign stage_is_id__linx_cpu_pyc__L95 = v245;
assign v246 = (state__stage == v168);
assign stage_is_ex__linx_cpu_pyc__L96 = v246;
assign v247 = (state__stage == v167);
assign stage_is_mem__linx_cpu_pyc__L97 = v247;
assign v248 = (state__stage == v166);
assign stage_is_wb__linx_cpu_pyc__L98 = v248;
assign v249 = (~state__halted);
assign v250 = (stage_is_wb__linx_cpu_pyc__L98 & v249);
assign v251 = (memwb__op == v165);
assign v252 = (memwb__op == v175);
assign v253 = (v251 | v252);
assign v254 = (v250 & v253);
assign v255 = v254;
assign halt_set__linx_cpu_pyc__L100 = v255;
pyc_or #(.WIDTH(1)) v256_inst (
  .a(state__halted),
  .b(halt_set__linx_cpu_pyc__L100),
  .y(v256)
);
assign stop__linx_cpu_pyc__L101 = v256;
pyc_not #(.WIDTH(1)) v257_inst (
  .a(stop__linx_cpu_pyc__L101),
  .y(v257)
);
assign active__linx_cpu_pyc__L102 = v257;
pyc_and #(.WIDTH(1)) v258_inst (
  .a(stage_is_if__linx_cpu_pyc__L94),
  .b(active__linx_cpu_pyc__L102),
  .y(v258)
);
assign do_if__linx_cpu_pyc__L104 = v258;
pyc_and #(.WIDTH(1)) v259_inst (
  .a(stage_is_id__linx_cpu_pyc__L95),
  .b(active__linx_cpu_pyc__L102),
  .y(v259)
);
assign do_id__linx_cpu_pyc__L105 = v259;
pyc_and #(.WIDTH(1)) v260_inst (
  .a(stage_is_ex__linx_cpu_pyc__L96),
  .b(active__linx_cpu_pyc__L102),
  .y(v260)
);
assign do_ex__linx_cpu_pyc__L106 = v260;
pyc_and #(.WIDTH(1)) v261_inst (
  .a(stage_is_mem__linx_cpu_pyc__L97),
  .b(active__linx_cpu_pyc__L102),
  .y(v261)
);
assign do_mem__linx_cpu_pyc__L107 = v261;
pyc_and #(.WIDTH(1)) v262_inst (
  .a(stage_is_wb__linx_cpu_pyc__L98),
  .b(active__linx_cpu_pyc__L102),
  .y(v262)
);
assign do_wb__linx_cpu_pyc__L108 = v262;
assign v263 = (v261 & exmem__is_load);
assign v264 = (v263 ? exmem__addr : v173);
assign v265 = (do_if__linx_cpu_pyc__L104 ? state__pc : v264);
assign v266 = v265;
assign mem_raddr__linx_cpu_pyc__L111 = v266;
pyc_and #(.WIDTH(1)) v267_inst (
  .a(v261),
  .b(exmem__is_store),
  .y(v267)
);
assign mem_wvalid__linx_cpu_pyc__L112 = v267;
assign mem_waddr__linx_cpu_pyc__L113 = exmem__addr;
assign mem_wdata__linx_cpu_pyc__L114 = exmem__wdata;
assign v268 = (exmem__size == v176);
assign v269 = (v268 ? v164 : v174);
assign v270 = v269;
assign mem_wstrb__linx_cpu_pyc__L115 = v270;
assign v271 = (exmem__size == v166);
assign v272 = (v271 ? v163 : mem_wstrb__linx_cpu_pyc__L115);
assign v273 = v272;
assign mem_wstrb__linx_cpu_pyc__L116 = v273;
pyc_byte_mem #(.ADDR_WIDTH(64), .DATA_WIDTH(64), .DEPTH(1048576)) mem (
  .clk(clk),
  .rst(rst),
  .raddr(mem_raddr__linx_cpu_pyc__L111),
  .rdata(v274),
  .wvalid(mem_wvalid__linx_cpu_pyc__L112),
  .waddr(mem_waddr__linx_cpu_pyc__L113),
  .wdata(mem_wdata__linx_cpu_pyc__L114),
  .wstrb(mem_wstrb__linx_cpu_pyc__L116)
);
assign mem_rdata__linx_cpu_pyc__L118 = v274;
pyc_mux #(.WIDTH(64)) v275_inst (
  .sel(do_if__linx_cpu_pyc__L104),
  .a(mem_rdata__linx_cpu_pyc__L118),
  .b(ifid__window),
  .y(v275)
);
assign ifid__window__next = v275;
assign v276 = ifid__window[15:0];
assign v277 = ifid__window[31:0];
assign v278 = ifid__window[47:0];
assign v279 = v276[3:0];
assign v280 = (v279 == v162);
assign v281 = v276[0];
assign v282 = (~v280);
assign v283 = (v282 & v281);
assign v284 = (~v281);
assign v285 = (v282 & v284);
assign v286 = v277[11:7];
assign v287 = {{1{1'b0}}, v286};
assign v288 = v277[19:15];
assign v289 = {{1{1'b0}}, v288};
assign v290 = v277[24:20];
assign v291 = {{1{1'b0}}, v290};
assign v292 = v277[31:27];
assign v293 = {{1{1'b0}}, v292};
assign v294 = v277[31:20];
assign v295 = {{52{1'b0}}, v294};
assign v296 = {{52{v294[11]}}, v294};
assign v297 = v277[31:12];
assign v298 = {{44{v297[19]}}, v297};
assign v299 = v277[31:25];
assign v300 = {{7{1'b0}}, v286};
assign v301 = (v300 << 7);
assign v302 = {{5{1'b0}}, v299};
assign v303 = (v301 | v302);
assign v304 = {{52{v303[11]}}, v303};
assign v305 = v277[31:15];
assign v306 = {{47{v305[16]}}, v305};
assign v307 = v278[15:0];
assign v308 = v278[47:16];
assign v309 = v307[15:4];
assign v310 = v308[31:12];
assign v311 = {{20{1'b0}}, v309};
assign v312 = (v311 << 20);
assign v313 = {{12{1'b0}}, v310};
assign v314 = (v312 | v313);
assign v315 = {{32{v314[31]}}, v314};
assign v316 = v308[11:7];
assign v317 = {{1{1'b0}}, v316};
assign v318 = v276[15:11];
assign v319 = {{1{1'b0}}, v318};
assign v320 = v276[10:6];
assign v321 = {{1{1'b0}}, v320};
assign v322 = {{59{v318[4]}}, v318};
assign v323 = {{59{v320[4]}}, v320};
assign v324 = v276[15:4];
assign v325 = {{52{v324[11]}}, v324};
assign v326 = {{59{1'b0}}, v320};
assign v327 = v276[13:11];
assign v328 = {{61{1'b0}}, v327};
assign v329 = (v276 & v161);
assign v330 = (v329 == v160);
assign v331 = (v285 & v330);
assign v332 = (v331 ? v159 : v175);
assign v333 = (v331 ? v168 : v176);
assign v334 = (v331 ? v319 : v170);
assign v335 = (v331 ? v170 : v170);
assign v336 = (v331 ? v323 : v173);
assign v337 = (v276 & v158);
assign v338 = (v337 == v157);
assign v339 = (v285 & v338);
assign v340 = (v326 << 1);
assign v341 = (v339 ? v155 : v332);
assign v342 = (v339 ? v168 : v333);
assign v343 = (v339 ? v156 : v334);
assign v344 = (v339 ? v170 : v335);
assign v345 = (v339 ? v340 : v336);
assign v346 = (v329 == v154);
assign v347 = (v285 & v346);
assign v348 = (v347 ? v152 : v341);
assign v349 = (v347 ? v168 : v342);
assign v350 = (v347 ? v170 : v343);
assign v351 = (v347 ? v321 : v344);
assign v352 = (v347 ? v153 : v344);
assign v353 = (v347 ? v170 : v344);
assign v354 = (v347 ? v322 : v345);
assign v355 = (v329 == v151);
assign v356 = (v285 & v355);
assign v357 = (v356 ? v150 : v348);
assign v358 = (v356 ? v168 : v349);
assign v359 = (v356 ? v170 : v350);
assign v360 = (v356 ? v321 : v351);
assign v361 = (v356 ? v170 : v352);
assign v362 = (v356 ? v170 : v353);
assign v363 = (v356 ? v322 : v354);
assign v364 = (v329 == v149);
assign v365 = (v285 & v364);
assign v366 = (v365 ? v148 : v357);
assign v367 = (v365 ? v168 : v358);
assign v368 = (v365 ? v319 : v359);
assign v369 = (v365 ? v321 : v360);
assign v370 = (v365 ? v170 : v361);
assign v371 = (v365 ? v170 : v362);
assign v372 = (v365 ? v173 : v363);
assign v373 = (v329 == v147);
assign v374 = (v285 & v373);
assign v375 = (v374 ? v146 : v366);
assign v376 = (v374 ? v168 : v367);
assign v377 = (v374 ? v170 : v368);
assign v378 = (v374 ? v321 : v369);
assign v379 = (v374 ? v319 : v370);
assign v380 = (v374 ? v170 : v371);
assign v381 = (v374 ? v173 : v372);
assign v382 = (v337 == v145);
assign v383 = (v285 & v382);
assign v384 = (v383 ? v153 : v375);
assign v385 = (v383 ? v168 : v376);
assign v386 = (v383 ? v170 : v377);
assign v387 = (v383 ? v321 : v378);
assign v388 = (v383 ? v170 : v379);
assign v389 = (v383 ? v170 : v380);
assign v390 = (v383 ? v173 : v381);
assign v391 = (v276 & v144);
assign v392 = (v391 == v143);
assign v393 = (v285 & v392);
assign v394 = (v325 << 1);
assign v395 = (v393 ? v142 : v384);
assign v396 = (v393 ? v168 : v385);
assign v397 = (v393 ? v170 : v386);
assign v398 = (v393 ? v170 : v387);
assign v399 = (v393 ? v170 : v388);
assign v400 = (v393 ? v170 : v389);
assign v401 = (v393 ? v394 : v390);
assign v402 = (v276 & v141);
assign v403 = (v402 == v140);
assign v404 = (v285 & v403);
assign v405 = (v404 ? v139 : v395);
assign v406 = (v404 ? v168 : v396);
assign v407 = (v404 ? v170 : v397);
assign v408 = (v404 ? v170 : v398);
assign v409 = (v404 ? v170 : v399);
assign v410 = (v404 ? v170 : v400);
assign v411 = (v404 ? v328 : v401);
assign v412 = (v276 & v138);
assign v413 = (v412 == v140);
assign v414 = (v285 & v413);
assign v415 = (v414 ? v137 : v405);
assign v416 = (v414 ? v168 : v406);
assign v417 = (v414 ? v170 : v407);
assign v418 = (v414 ? v170 : v408);
assign v419 = (v414 ? v170 : v409);
assign v420 = (v414 ? v170 : v410);
assign v421 = (v414 ? v173 : v411);
assign v422 = (v277 & v136);
assign v423 = (v422 == v135);
assign v424 = (v283 & v423);
assign v425 = (v424 ? v134 : v415);
assign v426 = (v424 ? v166 : v416);
assign v427 = (v424 ? v287 : v417);
assign v428 = (v424 ? v289 : v418);
assign v429 = (v424 ? v291 : v419);
assign v430 = (v424 ? v170 : v420);
assign v431 = (v424 ? v173 : v421);
assign v432 = (v422 == v133);
assign v433 = (v283 & v432);
assign v434 = (v433 ? v132 : v425);
assign v435 = (v433 ? v166 : v426);
assign v436 = (v433 ? v287 : v427);
assign v437 = (v433 ? v289 : v428);
assign v438 = (v433 ? v291 : v429);
assign v439 = (v433 ? v170 : v430);
assign v440 = (v433 ? v173 : v431);
assign v441 = (v422 == v131);
assign v442 = (v283 & v441);
assign v443 = (v442 ? v130 : v434);
assign v444 = (v442 ? v166 : v435);
assign v445 = (v442 ? v287 : v436);
assign v446 = (v442 ? v289 : v437);
assign v447 = (v442 ? v291 : v438);
assign v448 = (v442 ? v170 : v439);
assign v449 = (v442 ? v173 : v440);
assign v450 = (v422 == v129);
assign v451 = (v283 & v450);
assign v452 = (v451 ? v128 : v443);
assign v453 = (v451 ? v166 : v444);
assign v454 = (v451 ? v287 : v445);
assign v455 = (v451 ? v289 : v446);
assign v456 = (v451 ? v291 : v447);
assign v457 = (v451 ? v170 : v448);
assign v458 = (v451 ? v173 : v449);
assign v459 = (v422 == v127);
assign v460 = (v283 & v459);
assign v461 = (v460 ? v126 : v452);
assign v462 = (v460 ? v166 : v453);
assign v463 = (v460 ? v287 : v454);
assign v464 = (v460 ? v289 : v455);
assign v465 = (v460 ? v291 : v456);
assign v466 = (v460 ? v293 : v457);
assign v467 = (v460 ? v173 : v458);
assign v468 = (v422 == v125);
assign v469 = (v283 & v468);
assign v470 = (v469 ? v156 : v461);
assign v471 = (v469 ? v166 : v462);
assign v472 = (v469 ? v170 : v463);
assign v473 = (v469 ? v289 : v464);
assign v474 = (v469 ? v291 : v465);
assign v475 = (v469 ? v170 : v466);
assign v476 = (v469 ? v304 : v467);
assign v477 = (v422 == v124);
assign v478 = (v283 & v477);
assign v479 = (v478 ? v123 : v470);
assign v480 = (v478 ? v166 : v471);
assign v481 = (v478 ? v170 : v472);
assign v482 = (v478 ? v289 : v473);
assign v483 = (v478 ? v291 : v474);
assign v484 = (v478 ? v170 : v475);
assign v485 = (v478 ? v304 : v476);
assign v486 = (v422 == v122);
assign v487 = (v283 & v486);
assign v488 = (v487 ? v121 : v479);
assign v489 = (v487 ? v166 : v480);
assign v490 = (v487 ? v287 : v481);
assign v491 = (v487 ? v289 : v482);
assign v492 = (v487 ? v170 : v483);
assign v493 = (v487 ? v170 : v484);
assign v494 = (v487 ? v296 : v485);
assign v495 = (v422 == v120);
assign v496 = (v283 & v495);
assign v497 = (v496 ? v119 : v488);
assign v498 = (v496 ? v166 : v489);
assign v499 = (v496 ? v287 : v490);
assign v500 = (v496 ? v289 : v491);
assign v501 = (v496 ? v170 : v492);
assign v502 = (v496 ? v170 : v493);
assign v503 = (v496 ? v295 : v494);
assign v504 = (v422 == v118);
assign v505 = (v283 & v504);
assign v506 = (v505 ? v117 : v497);
assign v507 = (v505 ? v166 : v498);
assign v508 = (v505 ? v287 : v499);
assign v509 = (v505 ? v289 : v500);
assign v510 = (v505 ? v170 : v501);
assign v511 = (v505 ? v170 : v502);
assign v512 = (v505 ? v295 : v503);
assign v513 = (v422 == v116);
assign v514 = (v283 & v513);
assign v515 = (v514 ? v115 : v506);
assign v516 = (v514 ? v166 : v507);
assign v517 = (v514 ? v287 : v508);
assign v518 = (v514 ? v289 : v509);
assign v519 = (v514 ? v170 : v510);
assign v520 = (v514 ? v170 : v511);
assign v521 = (v514 ? v295 : v512);
assign v522 = (v277 & v114);
assign v523 = (v522 == v113);
assign v524 = (v283 & v523);
assign v525 = (v524 ? v112 : v515);
assign v526 = (v524 ? v166 : v516);
assign v527 = (v524 ? v287 : v517);
assign v528 = (v524 ? v289 : v518);
assign v529 = (v524 ? v291 : v519);
assign v530 = (v524 ? v170 : v520);
assign v531 = (v524 ? v173 : v521);
assign v532 = (v277 & v111);
assign v533 = (v532 == v110);
assign v534 = (v283 & v533);
assign v535 = (v534 ? v165 : v525);
assign v536 = (v534 ? v166 : v526);
assign v537 = (v534 ? v170 : v527);
assign v538 = (v534 ? v170 : v528);
assign v539 = (v534 ? v170 : v529);
assign v540 = (v534 ? v170 : v530);
assign v541 = (v534 ? v173 : v531);
assign v542 = (v277 & v109);
assign v543 = (v542 == v108);
assign v544 = (v283 & v543);
assign v545 = (v298 << 12);
assign v546 = (v544 ? v107 : v535);
assign v547 = (v544 ? v166 : v536);
assign v548 = (v544 ? v287 : v537);
assign v549 = (v544 ? v170 : v538);
assign v550 = (v544 ? v170 : v539);
assign v551 = (v544 ? v170 : v540);
assign v552 = (v544 ? v545 : v541);
assign v553 = (v277 & v106);
assign v554 = (v553 == v105);
assign v555 = (v283 & v554);
assign v556 = (v306 << 1);
assign v557 = (v555 ? v104 : v546);
assign v558 = (v555 ? v166 : v547);
assign v559 = (v555 ? v170 : v548);
assign v560 = (v555 ? v170 : v549);
assign v561 = (v555 ? v170 : v550);
assign v562 = (v555 ? v170 : v551);
assign v563 = (v555 ? v556 : v552);
assign v564 = (v278 & v103);
assign v565 = (v564 == v102);
assign v566 = (v280 & v565);
assign v567 = (v566 ? v101 : v557);
assign v568 = (v566 ? v100 : v558);
assign v569 = (v566 ? v317 : v559);
assign v570 = (v566 ? v170 : v560);
assign v571 = (v566 ? v170 : v561);
assign v572 = (v566 ? v170 : v562);
assign v573 = (v566 ? v315 : v563);
assign v574 = (do_id__linx_cpu_pyc__L105 ? v567 : idex__op);
assign v575 = v568;
assign v576 = v569;
assign v577 = v570;
assign v578 = v571;
assign v579 = v572;
assign v580 = v573;
assign v581 = v574;
assign idex__op__next = v581;
pyc_mux #(.WIDTH(3)) v582_inst (
  .sel(do_id__linx_cpu_pyc__L105),
  .a(v575),
  .b(idex__len_bytes),
  .y(v582)
);
assign idex__len_bytes__next = v582;
pyc_mux #(.WIDTH(6)) v583_inst (
  .sel(do_id__linx_cpu_pyc__L105),
  .a(v576),
  .b(idex__regdst),
  .y(v583)
);
assign idex__regdst__next = v583;
pyc_mux #(.WIDTH(6)) v584_inst (
  .sel(do_id__linx_cpu_pyc__L105),
  .a(v577),
  .b(idex__srcl),
  .y(v584)
);
assign idex__srcl__next = v584;
pyc_mux #(.WIDTH(6)) v585_inst (
  .sel(do_id__linx_cpu_pyc__L105),
  .a(v578),
  .b(idex__srcr),
  .y(v585)
);
assign idex__srcr__next = v585;
pyc_mux #(.WIDTH(6)) v586_inst (
  .sel(do_id__linx_cpu_pyc__L105),
  .a(v579),
  .b(idex__srcp),
  .y(v586)
);
assign idex__srcp__next = v586;
pyc_mux #(.WIDTH(64)) v587_inst (
  .sel(do_id__linx_cpu_pyc__L105),
  .a(v580),
  .b(idex__imm),
  .y(v587)
);
assign idex__imm__next = v587;
assign v588 = (v577 == v175);
assign v589 = (v588 ? v173 : v173);
assign v590 = (v577 == v139);
assign v591 = (v590 ? gpr__r1 : v589);
assign v592 = (v577 == v137);
assign v593 = (v592 ? gpr__r2 : v591);
assign v594 = (v577 == v148);
assign v595 = (v594 ? gpr__r3 : v593);
assign v596 = (v577 == v150);
assign v597 = (v596 ? gpr__r4 : v595);
assign v598 = (v577 == v152);
assign v599 = (v598 ? gpr__r5 : v597);
assign v600 = (v577 == v115);
assign v601 = (v600 ? gpr__r6 : v599);
assign v602 = (v577 == v117);
assign v603 = (v602 ? gpr__r7 : v601);
assign v604 = (v577 == v119);
assign v605 = (v604 ? gpr__r8 : v603);
assign v606 = (v577 == v121);
assign v607 = (v606 ? gpr__r9 : v605);
assign v608 = (v577 == v156);
assign v609 = (v608 ? gpr__r10 : v607);
assign v610 = (v577 == v128);
assign v611 = (v610 ? gpr__r11 : v609);
assign v612 = (v577 == v130);
assign v613 = (v612 ? gpr__r12 : v611);
assign v614 = (v577 == v132);
assign v615 = (v614 ? gpr__r13 : v613);
assign v616 = (v577 == v134);
assign v617 = (v616 ? gpr__r14 : v615);
assign v618 = (v577 == v112);
assign v619 = (v618 ? gpr__r15 : v617);
assign v620 = (v577 == v126);
assign v621 = (v620 ? gpr__r16 : v619);
assign v622 = (v577 == v101);
assign v623 = (v622 ? gpr__r17 : v621);
assign v624 = (v577 == v165);
assign v625 = (v624 ? gpr__r18 : v623);
assign v626 = (v577 == v142);
assign v627 = (v626 ? gpr__r19 : v625);
assign v628 = (v577 == v104);
assign v629 = (v628 ? gpr__r20 : v627);
assign v630 = (v577 == v159);
assign v631 = (v630 ? gpr__r21 : v629);
assign v632 = (v577 == v155);
assign v633 = (v632 ? gpr__r22 : v631);
assign v634 = (v577 == v146);
assign v635 = (v634 ? gpr__r23 : v633);
assign v636 = (v577 == v153);
assign v637 = (v636 ? t__r0 : v635);
assign v638 = (v577 == v107);
assign v639 = (v638 ? t__r1 : v637);
assign v640 = (v577 == v123);
assign v641 = (v640 ? t__r2 : v639);
assign v642 = (v577 == v99);
assign v643 = (v642 ? t__r3 : v641);
assign v644 = (v577 == v98);
assign v645 = (v644 ? u__r0 : v643);
assign v646 = (v577 == v97);
assign v647 = (v646 ? u__r1 : v645);
assign v648 = (v577 == v96);
assign v649 = (v648 ? u__r2 : v647);
assign v650 = (v577 == v95);
assign v651 = (v650 ? u__r3 : v649);
assign v652 = (v578 == v175);
assign v653 = (v652 ? v173 : v173);
assign v654 = (v578 == v139);
assign v655 = (v654 ? gpr__r1 : v653);
assign v656 = (v578 == v137);
assign v657 = (v656 ? gpr__r2 : v655);
assign v658 = (v578 == v148);
assign v659 = (v658 ? gpr__r3 : v657);
assign v660 = (v578 == v150);
assign v661 = (v660 ? gpr__r4 : v659);
assign v662 = (v578 == v152);
assign v663 = (v662 ? gpr__r5 : v661);
assign v664 = (v578 == v115);
assign v665 = (v664 ? gpr__r6 : v663);
assign v666 = (v578 == v117);
assign v667 = (v666 ? gpr__r7 : v665);
assign v668 = (v578 == v119);
assign v669 = (v668 ? gpr__r8 : v667);
assign v670 = (v578 == v121);
assign v671 = (v670 ? gpr__r9 : v669);
assign v672 = (v578 == v156);
assign v673 = (v672 ? gpr__r10 : v671);
assign v674 = (v578 == v128);
assign v675 = (v674 ? gpr__r11 : v673);
assign v676 = (v578 == v130);
assign v677 = (v676 ? gpr__r12 : v675);
assign v678 = (v578 == v132);
assign v679 = (v678 ? gpr__r13 : v677);
assign v680 = (v578 == v134);
assign v681 = (v680 ? gpr__r14 : v679);
assign v682 = (v578 == v112);
assign v683 = (v682 ? gpr__r15 : v681);
assign v684 = (v578 == v126);
assign v685 = (v684 ? gpr__r16 : v683);
assign v686 = (v578 == v101);
assign v687 = (v686 ? gpr__r17 : v685);
assign v688 = (v578 == v165);
assign v689 = (v688 ? gpr__r18 : v687);
assign v690 = (v578 == v142);
assign v691 = (v690 ? gpr__r19 : v689);
assign v692 = (v578 == v104);
assign v693 = (v692 ? gpr__r20 : v691);
assign v694 = (v578 == v159);
assign v695 = (v694 ? gpr__r21 : v693);
assign v696 = (v578 == v155);
assign v697 = (v696 ? gpr__r22 : v695);
assign v698 = (v578 == v146);
assign v699 = (v698 ? gpr__r23 : v697);
assign v700 = (v578 == v153);
assign v701 = (v700 ? t__r0 : v699);
assign v702 = (v578 == v107);
assign v703 = (v702 ? t__r1 : v701);
assign v704 = (v578 == v123);
assign v705 = (v704 ? t__r2 : v703);
assign v706 = (v578 == v99);
assign v707 = (v706 ? t__r3 : v705);
assign v708 = (v578 == v98);
assign v709 = (v708 ? u__r0 : v707);
assign v710 = (v578 == v97);
assign v711 = (v710 ? u__r1 : v709);
assign v712 = (v578 == v96);
assign v713 = (v712 ? u__r2 : v711);
assign v714 = (v578 == v95);
assign v715 = (v714 ? u__r3 : v713);
assign v716 = (v579 == v175);
assign v717 = (v716 ? v173 : v173);
assign v718 = (v579 == v139);
assign v719 = (v718 ? gpr__r1 : v717);
assign v720 = (v579 == v137);
assign v721 = (v720 ? gpr__r2 : v719);
assign v722 = (v579 == v148);
assign v723 = (v722 ? gpr__r3 : v721);
assign v724 = (v579 == v150);
assign v725 = (v724 ? gpr__r4 : v723);
assign v726 = (v579 == v152);
assign v727 = (v726 ? gpr__r5 : v725);
assign v728 = (v579 == v115);
assign v729 = (v728 ? gpr__r6 : v727);
assign v730 = (v579 == v117);
assign v731 = (v730 ? gpr__r7 : v729);
assign v732 = (v579 == v119);
assign v733 = (v732 ? gpr__r8 : v731);
assign v734 = (v579 == v121);
assign v735 = (v734 ? gpr__r9 : v733);
assign v736 = (v579 == v156);
assign v737 = (v736 ? gpr__r10 : v735);
assign v738 = (v579 == v128);
assign v739 = (v738 ? gpr__r11 : v737);
assign v740 = (v579 == v130);
assign v741 = (v740 ? gpr__r12 : v739);
assign v742 = (v579 == v132);
assign v743 = (v742 ? gpr__r13 : v741);
assign v744 = (v579 == v134);
assign v745 = (v744 ? gpr__r14 : v743);
assign v746 = (v579 == v112);
assign v747 = (v746 ? gpr__r15 : v745);
assign v748 = (v579 == v126);
assign v749 = (v748 ? gpr__r16 : v747);
assign v750 = (v579 == v101);
assign v751 = (v750 ? gpr__r17 : v749);
assign v752 = (v579 == v165);
assign v753 = (v752 ? gpr__r18 : v751);
assign v754 = (v579 == v142);
assign v755 = (v754 ? gpr__r19 : v753);
assign v756 = (v579 == v104);
assign v757 = (v756 ? gpr__r20 : v755);
assign v758 = (v579 == v159);
assign v759 = (v758 ? gpr__r21 : v757);
assign v760 = (v579 == v155);
assign v761 = (v760 ? gpr__r22 : v759);
assign v762 = (v579 == v146);
assign v763 = (v762 ? gpr__r23 : v761);
assign v764 = (v579 == v153);
assign v765 = (v764 ? t__r0 : v763);
assign v766 = (v579 == v107);
assign v767 = (v766 ? t__r1 : v765);
assign v768 = (v579 == v123);
assign v769 = (v768 ? t__r2 : v767);
assign v770 = (v579 == v99);
assign v771 = (v770 ? t__r3 : v769);
assign v772 = (v579 == v98);
assign v773 = (v772 ? u__r0 : v771);
assign v774 = (v579 == v97);
assign v775 = (v774 ? u__r1 : v773);
assign v776 = (v579 == v96);
assign v777 = (v776 ? u__r2 : v775);
assign v778 = (v579 == v95);
assign v779 = (v778 ? u__r3 : v777);
assign v780 = (do_id__linx_cpu_pyc__L105 ? v651 : idex__srcl_val);
assign v781 = v715;
assign v782 = v779;
assign v783 = v780;
assign idex__srcl_val__next = v783;
pyc_mux #(.WIDTH(64)) v784_inst (
  .sel(do_id__linx_cpu_pyc__L105),
  .a(v781),
  .b(idex__srcr_val),
  .y(v784)
);
assign idex__srcr_val__next = v784;
pyc_mux #(.WIDTH(64)) v785_inst (
  .sel(do_id__linx_cpu_pyc__L105),
  .a(v782),
  .b(idex__srcp_val),
  .y(v785)
);
assign idex__srcp_val__next = v785;
assign v786 = (idex__op == v139);
assign v787 = (idex__op == v142);
assign v788 = (idex__op == v104);
assign v789 = (idex__op == v148);
assign v790 = (idex__op == v159);
assign v791 = (idex__op == v155);
assign v792 = (idex__op == v146);
assign v793 = (idex__op == v153);
assign v794 = (idex__op == v107);
assign v795 = (idex__op == v117);
assign v796 = (idex__op == v115);
assign v797 = (idex__op == v119);
assign v798 = (idex__op == v128);
assign v799 = (idex__op == v130);
assign v800 = (idex__op == v132);
assign v801 = (idex__op == v134);
assign v802 = (idex__op == v112);
assign v803 = (idex__op == v126);
assign v804 = (idex__op == v101);
assign v805 = (idex__op == v121);
assign v806 = (idex__op == v150);
assign v807 = (idex__op == v156);
assign v808 = (idex__op == v152);
assign v809 = (idex__op == v123);
assign v810 = (idex__imm << 2);
assign v811 = (v786 | v787);
assign v812 = (v811 | v788);
assign v813 = (v812 ? idex__imm : v173);
assign v814 = (v812 ? v177 : v177);
assign v815 = (v812 ? v176 : v176);
assign v816 = (v812 ? v173 : v173);
assign v817 = (v789 ? idex__srcl_val : v813);
assign v818 = (v789 ? v177 : v814);
assign v819 = (v789 ? v176 : v815);
assign v820 = (v789 ? v173 : v816);
assign v821 = (v790 ? idex__imm : v817);
assign v822 = (v790 ? v177 : v818);
assign v823 = (v790 ? v176 : v819);
assign v824 = (v790 ? v173 : v820);
assign v825 = (state__pc + idex__imm);
assign v826 = (v791 ? v825 : v821);
assign v827 = (v791 ? v177 : v822);
assign v828 = (v791 ? v176 : v823);
assign v829 = (v791 ? v173 : v824);
assign v830 = (idex__srcl_val == idex__srcr_val);
assign v831 = (v830 ? v172 : v173);
assign v832 = (v792 ? v831 : v826);
assign v833 = (v792 ? v177 : v827);
assign v834 = (v792 ? v176 : v828);
assign v835 = (v792 ? v173 : v829);
assign v836 = (v793 ? idex__srcl_val : v832);
assign v837 = (v793 ? v177 : v833);
assign v838 = (v793 ? v176 : v834);
assign v839 = (v793 ? v173 : v835);
assign v840 = (state__pc & v94);
assign v841 = (v840 + idex__imm);
assign v842 = (v794 ? v841 : v836);
assign v843 = (v794 ? v177 : v837);
assign v844 = (v794 ? v176 : v838);
assign v845 = (v794 ? v173 : v839);
assign v846 = (idex__srcl_val + idex__imm);
assign v847 = (v795 ? v846 : v842);
assign v848 = (v795 ? v177 : v843);
assign v849 = (v795 ? v176 : v844);
assign v850 = (v795 ? v173 : v845);
assign v851 = (~idex__imm);
assign v852 = (v851 + v172);
assign v853 = (idex__srcl_val + v852);
assign v854 = (v796 ? v853 : v847);
assign v855 = (v796 ? v177 : v848);
assign v856 = (v796 ? v176 : v849);
assign v857 = (v796 ? v173 : v850);
assign v858 = idex__srcl_val[31:0];
assign v859 = idex__imm[31:0];
assign v860 = (v858 + v859);
assign v861 = {{32{v860[31]}}, v860};
assign v862 = (v797 ? v861 : v854);
assign v863 = (v797 ? v177 : v855);
assign v864 = (v797 ? v176 : v856);
assign v865 = (v797 ? v173 : v857);
assign v866 = idex__srcr_val[31:0];
assign v867 = (v858 + v866);
assign v868 = {{32{v867[31]}}, v867};
assign v869 = (v858 | v866);
assign v870 = {{32{v869[31]}}, v869};
assign v871 = (v858 & v866);
assign v872 = {{32{v871[31]}}, v871};
assign v873 = (v858 ^ v866);
assign v874 = {{32{v873[31]}}, v873};
assign v875 = (v798 ? v868 : v862);
assign v876 = (v798 ? v177 : v863);
assign v877 = (v798 ? v176 : v864);
assign v878 = (v798 ? v173 : v865);
assign v879 = (v799 ? v870 : v875);
assign v880 = (v799 ? v177 : v876);
assign v881 = (v799 ? v176 : v877);
assign v882 = (v799 ? v173 : v878);
assign v883 = (v800 ? v872 : v879);
assign v884 = (v800 ? v177 : v880);
assign v885 = (v800 ? v176 : v881);
assign v886 = (v800 ? v173 : v882);
assign v887 = (v801 ? v874 : v883);
assign v888 = (v801 ? v177 : v884);
assign v889 = (v801 ? v176 : v885);
assign v890 = (v801 ? v173 : v886);
assign v891 = (v802 ? v831 : v887);
assign v892 = (v802 ? v177 : v888);
assign v893 = (v802 ? v176 : v889);
assign v894 = (v802 ? v173 : v890);
assign v895 = (v804 ? idex__imm : v891);
assign v896 = (v804 ? v177 : v892);
assign v897 = (v804 ? v176 : v893);
assign v898 = (v804 ? v173 : v894);
assign v899 = (idex__srcp_val == v173);
assign v900 = (~v899);
assign v901 = (v900 ? idex__srcr_val : idex__srcl_val);
assign v902 = (v803 ? v901 : v895);
assign v903 = (v803 ? v177 : v896);
assign v904 = (v803 ? v176 : v897);
assign v905 = (v803 ? v173 : v898);
assign v906 = (v805 | v806);
assign v907 = (idex__srcl_val + v810);
assign v908 = (v906 ? v173 : v902);
assign v909 = (v906 ? v178 : v903);
assign v910 = (v906 ? v177 : v903);
assign v911 = (v906 ? v166 : v904);
assign v912 = (v906 ? v907 : v905);
assign v913 = (v906 ? v173 : v905);
assign v914 = (idex__srcr_val + v810);
assign v915 = (v807 ? v914 : v907);
assign v916 = (v807 ? idex__srcl_val : idex__srcr_val);
assign v917 = (v807 | v808);
assign v918 = (v917 ? v173 : v908);
assign v919 = (v917 ? v177 : v909);
assign v920 = (v917 ? v178 : v910);
assign v921 = (v917 ? v166 : v911);
assign v922 = (v917 ? v915 : v912);
assign v923 = (v917 ? v916 : v913);
assign v924 = (idex__imm << 3);
assign v925 = (idex__srcr_val + v924);
assign v926 = (v809 ? v173 : v918);
assign v927 = (v809 ? v177 : v919);
assign v928 = (v809 ? v178 : v920);
assign v929 = (v809 ? v176 : v921);
assign v930 = (v809 ? v925 : v922);
assign v931 = (v809 ? idex__srcl_val : v923);
assign v932 = (do_ex__linx_cpu_pyc__L106 ? idex__op : exmem__op);
assign v933 = v926;
assign v934 = v927;
assign v935 = v928;
assign v936 = v929;
assign v937 = v930;
assign v938 = v931;
assign v939 = v932;
assign exmem__op__next = v939;
pyc_mux #(.WIDTH(3)) v940_inst (
  .sel(do_ex__linx_cpu_pyc__L106),
  .a(idex__len_bytes),
  .b(exmem__len_bytes),
  .y(v940)
);
assign exmem__len_bytes__next = v940;
pyc_mux #(.WIDTH(6)) v941_inst (
  .sel(do_ex__linx_cpu_pyc__L106),
  .a(idex__regdst),
  .b(exmem__regdst),
  .y(v941)
);
assign exmem__regdst__next = v941;
pyc_mux #(.WIDTH(64)) v942_inst (
  .sel(do_ex__linx_cpu_pyc__L106),
  .a(v933),
  .b(exmem__alu),
  .y(v942)
);
assign exmem__alu__next = v942;
pyc_mux #(.WIDTH(1)) v943_inst (
  .sel(do_ex__linx_cpu_pyc__L106),
  .a(v934),
  .b(exmem__is_load),
  .y(v943)
);
assign exmem__is_load__next = v943;
pyc_mux #(.WIDTH(1)) v944_inst (
  .sel(do_ex__linx_cpu_pyc__L106),
  .a(v935),
  .b(exmem__is_store),
  .y(v944)
);
assign exmem__is_store__next = v944;
pyc_mux #(.WIDTH(3)) v945_inst (
  .sel(do_ex__linx_cpu_pyc__L106),
  .a(v936),
  .b(exmem__size),
  .y(v945)
);
assign exmem__size__next = v945;
pyc_mux #(.WIDTH(64)) v946_inst (
  .sel(do_ex__linx_cpu_pyc__L106),
  .a(v937),
  .b(exmem__addr),
  .y(v946)
);
assign exmem__addr__next = v946;
pyc_mux #(.WIDTH(64)) v947_inst (
  .sel(do_ex__linx_cpu_pyc__L106),
  .a(v938),
  .b(exmem__wdata),
  .y(v947)
);
assign exmem__wdata__next = v947;
assign v948 = mem_rdata__linx_cpu_pyc__L118[31:0];
assign v949 = {{32{v948[31]}}, v948};
assign v950 = (exmem__is_load ? v949 : exmem__alu);
assign v951 = (exmem__is_store ? v173 : v950);
assign v952 = (do_mem__linx_cpu_pyc__L107 ? exmem__op : memwb__op);
assign v953 = v951;
assign v954 = v952;
assign memwb__op__next = v954;
pyc_mux #(.WIDTH(3)) v955_inst (
  .sel(do_mem__linx_cpu_pyc__L107),
  .a(exmem__len_bytes),
  .b(memwb__len_bytes),
  .y(v955)
);
assign memwb__len_bytes__next = v955;
pyc_mux #(.WIDTH(6)) v956_inst (
  .sel(do_mem__linx_cpu_pyc__L107),
  .a(exmem__regdst),
  .b(memwb__regdst),
  .y(v956)
);
assign memwb__regdst__next = v956;
pyc_mux #(.WIDTH(64)) v957_inst (
  .sel(do_mem__linx_cpu_pyc__L107),
  .a(v953),
  .b(memwb__value),
  .y(v957)
);
assign memwb__value__next = v957;
pyc_mux #(.WIDTH(1)) v958_inst (
  .sel(halt_set__linx_cpu_pyc__L100),
  .a(v178),
  .b(state__halted),
  .y(v958)
);
assign state__halted__next = v958;
assign v959 = (memwb__op == v139);
assign v960 = (memwb__op == v142);
assign v961 = (memwb__op == v104);
assign v962 = (memwb__op == v137);
assign v963 = (v959 | v960);
assign v964 = (v963 | v961);
assign v965 = (v964 | v962);
assign v966 = (state__br_kind == v93);
assign v967 = (state__br_kind == v92);
assign v968 = (state__br_kind == v91);
assign v969 = (state__br_base_pc + state__br_off);
assign v970 = (v968 ? state__commit_tgt : v969);
assign v971 = (v967 | v968);
assign v972 = (v966 & state__commit_cond);
assign v973 = (v971 | v972);
assign v974 = {{61{1'b0}}, memwb__len_bytes};
assign v975 = (state__pc + v974);
assign v976 = (v973 ? v970 : v975);
assign v977 = (v965 ? v976 : v975);
assign v978 = (do_wb__linx_cpu_pyc__L108 ? v977 : state__pc);
assign v979 = v959;
assign v980 = v960;
assign v981 = v961;
assign v982 = v962;
assign v983 = v964;
assign v984 = v965;
assign v985 = v973;
assign v986 = v978;
assign state__pc__next = v986;
assign v987 = (stage_is_if__linx_cpu_pyc__L94 ? v169 : state__stage);
assign v988 = (stage_is_id__linx_cpu_pyc__L95 ? v168 : v987);
assign v989 = (stage_is_ex__linx_cpu_pyc__L96 ? v167 : v988);
assign v990 = (stage_is_mem__linx_cpu_pyc__L97 ? v166 : v989);
assign v991 = (stage_is_wb__linx_cpu_pyc__L98 ? v176 : v990);
assign v992 = (stop__linx_cpu_pyc__L101 ? state__stage : v991);
assign v993 = v992;
assign state__stage__next = v993;
pyc_add #(.WIDTH(64)) v994_inst (
  .a(state__cycles),
  .b(v172),
  .y(v994)
);
assign state__cycles__next = v994;
assign v995 = (memwb__op == v146);
assign v996 = (memwb__op == v153);
assign v997 = (do_wb__linx_cpu_pyc__L108 & v984);
assign v998 = (v997 ? v177 : state__commit_cond);
assign v999 = (v997 ? v173 : state__commit_tgt);
assign v1000 = (do_wb__linx_cpu_pyc__L108 & v995);
assign v1001 = memwb__value[0];
assign v1002 = (v1000 ? v1001 : v998);
assign v1003 = (do_wb__linx_cpu_pyc__L108 & v996);
assign v1004 = (v1003 ? memwb__value : v999);
assign v1005 = v997;
assign v1006 = v1002;
assign v1007 = v1004;
assign state__commit_cond__next = v1006;
assign state__commit_tgt__next = v1007;
assign v1008 = (v1005 & v985);
assign v1009 = (v1008 ? v171 : state__br_kind);
assign v1010 = (v1008 ? state__pc : state__br_base_pc);
assign v1011 = (v1008 ? v173 : state__br_off);
assign v1012 = (do_wb__linx_cpu_pyc__L108 & v983);
assign v1013 = (~v985);
assign v1014 = (v1012 & v1013);
assign v1015 = (v1014 & v980);
assign v1016 = (v1015 ? v93 : v1009);
assign v1017 = (v1015 ? state__pc : v1010);
assign v1018 = (v1015 ? memwb__value : v1011);
assign v1019 = (v1014 & v981);
assign v1020 = (v1019 ? v92 : v1016);
assign v1021 = (v1019 ? state__pc : v1017);
assign v1022 = (v1019 ? memwb__value : v1018);
assign v1023 = memwb__value[2:0];
assign v1024 = (v1023 == v90);
assign v1025 = (v1024 ? v91 : v171);
assign v1026 = (v1014 & v979);
assign v1027 = (v1026 ? v1025 : v1020);
assign v1028 = (v1026 ? state__pc : v1021);
assign v1029 = (v1026 ? v173 : v1022);
assign v1030 = (do_wb__linx_cpu_pyc__L108 & v982);
assign v1031 = (v1030 ? v171 : v1027);
assign v1032 = (v1030 ? state__pc : v1028);
assign v1033 = (v1030 ? v173 : v1029);
assign v1034 = v1012;
assign v1035 = v1031;
assign v1036 = v1032;
assign v1037 = v1033;
assign state__br_kind__next = v1035;
assign state__br_base_pc__next = v1036;
assign state__br_off__next = v1037;
assign v1038 = (memwb__op == v156);
assign v1039 = (memwb__op == v152);
assign v1040 = (v1038 | v1039);
assign v1041 = (~v1040);
assign v1042 = (do_wb__linx_cpu_pyc__L108 & v1041);
assign v1043 = (memwb__regdst == v170);
assign v1044 = (~v1043);
assign v1045 = (v1042 & v1044);
assign v1046 = (memwb__op == v150);
assign v1047 = (do_wb__linx_cpu_pyc__L108 & v1046);
assign v1048 = (memwb__regdst == v95);
assign v1049 = (v1045 & v1048);
assign v1050 = (v1047 | v1049);
assign v1051 = (memwb__regdst == v96);
assign v1052 = (v1045 & v1051);
assign v1053 = v1045;
assign v1054 = v1050;
assign v1055 = v1052;
assign gpr__r0__next = v173;
assign v1056 = (memwb__regdst == v139);
assign v1057 = (v1053 & v1056);
assign v1058 = (v1057 ? memwb__value : gpr__r1);
assign v1059 = v1058;
assign gpr__r1__next = v1059;
assign v1060 = (memwb__regdst == v137);
assign v1061 = (v1053 & v1060);
assign v1062 = (v1061 ? memwb__value : gpr__r2);
assign v1063 = v1062;
assign gpr__r2__next = v1063;
assign v1064 = (memwb__regdst == v148);
assign v1065 = (v1053 & v1064);
assign v1066 = (v1065 ? memwb__value : gpr__r3);
assign v1067 = v1066;
assign gpr__r3__next = v1067;
assign v1068 = (memwb__regdst == v150);
assign v1069 = (v1053 & v1068);
assign v1070 = (v1069 ? memwb__value : gpr__r4);
assign v1071 = v1070;
assign gpr__r4__next = v1071;
assign v1072 = (memwb__regdst == v152);
assign v1073 = (v1053 & v1072);
assign v1074 = (v1073 ? memwb__value : gpr__r5);
assign v1075 = v1074;
assign gpr__r5__next = v1075;
assign v1076 = (memwb__regdst == v115);
assign v1077 = (v1053 & v1076);
assign v1078 = (v1077 ? memwb__value : gpr__r6);
assign v1079 = v1078;
assign gpr__r6__next = v1079;
assign v1080 = (memwb__regdst == v117);
assign v1081 = (v1053 & v1080);
assign v1082 = (v1081 ? memwb__value : gpr__r7);
assign v1083 = v1082;
assign gpr__r7__next = v1083;
assign v1084 = (memwb__regdst == v119);
assign v1085 = (v1053 & v1084);
assign v1086 = (v1085 ? memwb__value : gpr__r8);
assign v1087 = v1086;
assign gpr__r8__next = v1087;
assign v1088 = (memwb__regdst == v121);
assign v1089 = (v1053 & v1088);
assign v1090 = (v1089 ? memwb__value : gpr__r9);
assign v1091 = v1090;
assign gpr__r9__next = v1091;
assign v1092 = (memwb__regdst == v156);
assign v1093 = (v1053 & v1092);
assign v1094 = (v1093 ? memwb__value : gpr__r10);
assign v1095 = v1094;
assign gpr__r10__next = v1095;
assign v1096 = (memwb__regdst == v128);
assign v1097 = (v1053 & v1096);
assign v1098 = (v1097 ? memwb__value : gpr__r11);
assign v1099 = v1098;
assign gpr__r11__next = v1099;
assign v1100 = (memwb__regdst == v130);
assign v1101 = (v1053 & v1100);
assign v1102 = (v1101 ? memwb__value : gpr__r12);
assign v1103 = v1102;
assign gpr__r12__next = v1103;
assign v1104 = (memwb__regdst == v132);
assign v1105 = (v1053 & v1104);
assign v1106 = (v1105 ? memwb__value : gpr__r13);
assign v1107 = v1106;
assign gpr__r13__next = v1107;
assign v1108 = (memwb__regdst == v134);
assign v1109 = (v1053 & v1108);
assign v1110 = (v1109 ? memwb__value : gpr__r14);
assign v1111 = v1110;
assign gpr__r14__next = v1111;
assign v1112 = (memwb__regdst == v112);
assign v1113 = (v1053 & v1112);
assign v1114 = (v1113 ? memwb__value : gpr__r15);
assign v1115 = v1114;
assign gpr__r15__next = v1115;
assign v1116 = (memwb__regdst == v126);
assign v1117 = (v1053 & v1116);
assign v1118 = (v1117 ? memwb__value : gpr__r16);
assign v1119 = v1118;
assign gpr__r16__next = v1119;
assign v1120 = (memwb__regdst == v101);
assign v1121 = (v1053 & v1120);
assign v1122 = (v1121 ? memwb__value : gpr__r17);
assign v1123 = v1122;
assign gpr__r17__next = v1123;
assign v1124 = (memwb__regdst == v165);
assign v1125 = (v1053 & v1124);
assign v1126 = (v1125 ? memwb__value : gpr__r18);
assign v1127 = v1126;
assign gpr__r18__next = v1127;
assign v1128 = (memwb__regdst == v142);
assign v1129 = (v1053 & v1128);
assign v1130 = (v1129 ? memwb__value : gpr__r19);
assign v1131 = v1130;
assign gpr__r19__next = v1131;
assign v1132 = (memwb__regdst == v104);
assign v1133 = (v1053 & v1132);
assign v1134 = (v1133 ? memwb__value : gpr__r20);
assign v1135 = v1134;
assign gpr__r20__next = v1135;
assign v1136 = (memwb__regdst == v159);
assign v1137 = (v1053 & v1136);
assign v1138 = (v1137 ? memwb__value : gpr__r21);
assign v1139 = v1138;
assign gpr__r21__next = v1139;
assign v1140 = (memwb__regdst == v155);
assign v1141 = (v1053 & v1140);
assign v1142 = (v1141 ? memwb__value : gpr__r22);
assign v1143 = v1142;
assign gpr__r22__next = v1143;
assign v1144 = (memwb__regdst == v146);
assign v1145 = (v1053 & v1144);
assign v1146 = (v1145 ? memwb__value : gpr__r23);
assign v1147 = v1146;
assign gpr__r23__next = v1147;
assign v1148 = (v1054 ? memwb__value : t__r0);
assign v1149 = (v1034 ? v173 : v1148);
assign v1150 = (v1054 ? t__r0 : t__r1);
assign v1151 = (v1034 ? v173 : v1150);
assign v1152 = (v1054 ? t__r1 : t__r2);
assign v1153 = (v1034 ? v173 : v1152);
assign v1154 = (v1054 ? t__r2 : t__r3);
assign v1155 = (v1034 ? v173 : v1154);
assign v1156 = (v1055 ? memwb__value : u__r0);
assign v1157 = (v1034 ? v173 : v1156);
assign v1158 = (v1055 ? u__r0 : u__r1);
assign v1159 = (v1034 ? v173 : v1158);
assign v1160 = (v1055 ? u__r1 : u__r2);
assign v1161 = (v1034 ? v173 : v1160);
assign v1162 = (v1055 ? u__r2 : u__r3);
assign v1163 = (v1034 ? v173 : v1162);
assign v1164 = v1149;
assign v1165 = v1151;
assign v1166 = v1153;
assign v1167 = v1155;
assign v1168 = v1157;
assign v1169 = v1159;
assign v1170 = v1161;
assign v1171 = v1163;
assign t__r0__next = v1164;
assign t__r1__next = v1165;
assign t__r2__next = v1166;
assign t__r3__next = v1167;
assign u__r0__next = v1168;
assign u__r1__next = v1169;
assign u__r2__next = v1170;
assign u__r3__next = v1171;
assign halted = state__halted;
assign pc = state__pc;
assign stage = state__stage;
assign cycles = state__cycles;
assign a0 = gpr__r2;
assign a1 = gpr__r3;
assign ra = gpr__r10;
assign sp = gpr__r1;
assign br_kind = state__br_kind;
assign if_window = ifid__window;
assign wb_op = memwb__op;
assign wb_regdst = memwb__regdst;
assign wb_value = memwb__value;
assign commit_cond = state__commit_cond;
assign commit_tgt = state__commit_tgt;

endmodule

