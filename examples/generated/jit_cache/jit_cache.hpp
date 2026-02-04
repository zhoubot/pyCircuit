// pyCircuit C++ emission (prototype)
#include <pyc/cpp/pyc_sim.hpp>

namespace pyc::gen {

struct JitCache {
  pyc::cpp::Wire<1> sys_clk{};
  pyc::cpp::Wire<1> sys_rst{};
  pyc::cpp::Wire<1> req_valid{};
  pyc::cpp::Wire<32> req_addr{};
  pyc::cpp::Wire<1> rsp_ready{};
  pyc::cpp::Wire<1> req_ready{};
  pyc::cpp::Wire<1> rsp_valid{};
  pyc::cpp::Wire<1> rsp_hit{};
  pyc::cpp::Wire<32> rsp_rdata{};

  pyc::cpp::Wire<33> v1{};
  pyc::cpp::Wire<3> v2{};
  pyc::cpp::Wire<3> v3{};
  pyc::cpp::Wire<3> v4{};
  pyc::cpp::Wire<3> v5{};
  pyc::cpp::Wire<3> v6{};
  pyc::cpp::Wire<3> v7{};
  pyc::cpp::Wire<3> v8{};
  pyc::cpp::Wire<3> v9{};
  pyc::cpp::Wire<27> v10{};
  pyc::cpp::Wire<1> v11{};
  pyc::cpp::Wire<4> v12{};
  pyc::cpp::Wire<32> v13{};
  pyc::cpp::Wire<1> v14{};
  pyc::cpp::Wire<33> v15{};
  pyc::cpp::Wire<3> v16{};
  pyc::cpp::Wire<3> v17{};
  pyc::cpp::Wire<3> v18{};
  pyc::cpp::Wire<3> v19{};
  pyc::cpp::Wire<3> v20{};
  pyc::cpp::Wire<3> v21{};
  pyc::cpp::Wire<3> v22{};
  pyc::cpp::Wire<3> v23{};
  pyc::cpp::Wire<27> v24{};
  pyc::cpp::Wire<1> v25{};
  pyc::cpp::Wire<4> v26{};
  pyc::cpp::Wire<32> v27{};
  pyc::cpp::Wire<1> v28{};
  pyc::cpp::Wire<1> req_valid__jit_cache__L166{};
  pyc::cpp::Wire<32> req_addr__jit_cache__L167{};
  pyc::cpp::Wire<1> rsp_ready__jit_cache__L168{};
  pyc::cpp::Wire<1> cache__wvalid0__jit_cache__L172{};
  pyc::cpp::Wire<32> cache__waddr0__jit_cache__L173{};
  pyc::cpp::Wire<32> cache__wdata0__jit_cache__L174{};
  pyc::cpp::Wire<4> cache__wstrb0__jit_cache__L175{};
  pyc::cpp::Wire<1> cache__req_q__in_valid{};
  pyc::cpp::Wire<32> cache__req_q__in_data{};
  pyc::cpp::Wire<1> cache__req_q__out_ready{};
  pyc::cpp::Wire<1> v29{};
  pyc::cpp::Wire<1> v30{};
  pyc::cpp::Wire<32> v31{};
  pyc::cpp::Wire<1> cache__rsp_q__in_valid{};
  pyc::cpp::Wire<33> cache__rsp_q__in_data{};
  pyc::cpp::Wire<1> cache__rsp_q__out_ready{};
  pyc::cpp::Wire<1> v32{};
  pyc::cpp::Wire<1> v33{};
  pyc::cpp::Wire<33> v34{};
  pyc::cpp::Wire<1> v35{};
  pyc::cpp::Wire<1> cache__rsp_hit__jit_cache__L183{};
  pyc::cpp::Wire<32> v36{};
  pyc::cpp::Wire<32> cache__rsp_rdata__jit_cache__L184{};
  pyc::cpp::Wire<1> cache__valid__s0__w0__next{};
  pyc::cpp::Wire<1> v37{};
  pyc::cpp::Wire<1> cache__valid__s0__w0{};
  pyc::cpp::Wire<1> cache__valid__s0__w1__next{};
  pyc::cpp::Wire<1> v38{};
  pyc::cpp::Wire<1> cache__valid__s0__w1{};
  pyc::cpp::Wire<1> cache__valid__s1__w0__next{};
  pyc::cpp::Wire<1> v39{};
  pyc::cpp::Wire<1> cache__valid__s1__w0{};
  pyc::cpp::Wire<1> cache__valid__s1__w1__next{};
  pyc::cpp::Wire<1> v40{};
  pyc::cpp::Wire<1> cache__valid__s1__w1{};
  pyc::cpp::Wire<1> cache__valid__s2__w0__next{};
  pyc::cpp::Wire<1> v41{};
  pyc::cpp::Wire<1> cache__valid__s2__w0{};
  pyc::cpp::Wire<1> cache__valid__s2__w1__next{};
  pyc::cpp::Wire<1> v42{};
  pyc::cpp::Wire<1> cache__valid__s2__w1{};
  pyc::cpp::Wire<1> cache__valid__s3__w0__next{};
  pyc::cpp::Wire<1> v43{};
  pyc::cpp::Wire<1> cache__valid__s3__w0{};
  pyc::cpp::Wire<1> cache__valid__s3__w1__next{};
  pyc::cpp::Wire<1> v44{};
  pyc::cpp::Wire<1> cache__valid__s3__w1{};
  pyc::cpp::Wire<1> cache__valid__s4__w0__next{};
  pyc::cpp::Wire<1> v45{};
  pyc::cpp::Wire<1> cache__valid__s4__w0{};
  pyc::cpp::Wire<1> cache__valid__s4__w1__next{};
  pyc::cpp::Wire<1> v46{};
  pyc::cpp::Wire<1> cache__valid__s4__w1{};
  pyc::cpp::Wire<1> cache__valid__s5__w0__next{};
  pyc::cpp::Wire<1> v47{};
  pyc::cpp::Wire<1> cache__valid__s5__w0{};
  pyc::cpp::Wire<1> cache__valid__s5__w1__next{};
  pyc::cpp::Wire<1> v48{};
  pyc::cpp::Wire<1> cache__valid__s5__w1{};
  pyc::cpp::Wire<1> cache__valid__s6__w0__next{};
  pyc::cpp::Wire<1> v49{};
  pyc::cpp::Wire<1> cache__valid__s6__w0{};
  pyc::cpp::Wire<1> cache__valid__s6__w1__next{};
  pyc::cpp::Wire<1> v50{};
  pyc::cpp::Wire<1> cache__valid__s6__w1{};
  pyc::cpp::Wire<1> cache__valid__s7__w0__next{};
  pyc::cpp::Wire<1> v51{};
  pyc::cpp::Wire<1> cache__valid__s7__w0{};
  pyc::cpp::Wire<1> cache__valid__s7__w1__next{};
  pyc::cpp::Wire<1> v52{};
  pyc::cpp::Wire<1> cache__valid__s7__w1{};
  pyc::cpp::Wire<27> cache__tag__s0__w0__next{};
  pyc::cpp::Wire<27> v53{};
  pyc::cpp::Wire<27> cache__tag__s0__w0{};
  pyc::cpp::Wire<27> cache__tag__s0__w1__next{};
  pyc::cpp::Wire<27> v54{};
  pyc::cpp::Wire<27> cache__tag__s0__w1{};
  pyc::cpp::Wire<27> cache__tag__s1__w0__next{};
  pyc::cpp::Wire<27> v55{};
  pyc::cpp::Wire<27> cache__tag__s1__w0{};
  pyc::cpp::Wire<27> cache__tag__s1__w1__next{};
  pyc::cpp::Wire<27> v56{};
  pyc::cpp::Wire<27> cache__tag__s1__w1{};
  pyc::cpp::Wire<27> cache__tag__s2__w0__next{};
  pyc::cpp::Wire<27> v57{};
  pyc::cpp::Wire<27> cache__tag__s2__w0{};
  pyc::cpp::Wire<27> cache__tag__s2__w1__next{};
  pyc::cpp::Wire<27> v58{};
  pyc::cpp::Wire<27> cache__tag__s2__w1{};
  pyc::cpp::Wire<27> cache__tag__s3__w0__next{};
  pyc::cpp::Wire<27> v59{};
  pyc::cpp::Wire<27> cache__tag__s3__w0{};
  pyc::cpp::Wire<27> cache__tag__s3__w1__next{};
  pyc::cpp::Wire<27> v60{};
  pyc::cpp::Wire<27> cache__tag__s3__w1{};
  pyc::cpp::Wire<27> cache__tag__s4__w0__next{};
  pyc::cpp::Wire<27> v61{};
  pyc::cpp::Wire<27> cache__tag__s4__w0{};
  pyc::cpp::Wire<27> cache__tag__s4__w1__next{};
  pyc::cpp::Wire<27> v62{};
  pyc::cpp::Wire<27> cache__tag__s4__w1{};
  pyc::cpp::Wire<27> cache__tag__s5__w0__next{};
  pyc::cpp::Wire<27> v63{};
  pyc::cpp::Wire<27> cache__tag__s5__w0{};
  pyc::cpp::Wire<27> cache__tag__s5__w1__next{};
  pyc::cpp::Wire<27> v64{};
  pyc::cpp::Wire<27> cache__tag__s5__w1{};
  pyc::cpp::Wire<27> cache__tag__s6__w0__next{};
  pyc::cpp::Wire<27> v65{};
  pyc::cpp::Wire<27> cache__tag__s6__w0{};
  pyc::cpp::Wire<27> cache__tag__s6__w1__next{};
  pyc::cpp::Wire<27> v66{};
  pyc::cpp::Wire<27> cache__tag__s6__w1{};
  pyc::cpp::Wire<27> cache__tag__s7__w0__next{};
  pyc::cpp::Wire<27> v67{};
  pyc::cpp::Wire<27> cache__tag__s7__w0{};
  pyc::cpp::Wire<27> cache__tag__s7__w1__next{};
  pyc::cpp::Wire<27> v68{};
  pyc::cpp::Wire<27> cache__tag__s7__w1{};
  pyc::cpp::Wire<32> cache__data__s0__w0__next{};
  pyc::cpp::Wire<32> v69{};
  pyc::cpp::Wire<32> cache__data__s0__w0{};
  pyc::cpp::Wire<32> cache__data__s0__w1__next{};
  pyc::cpp::Wire<32> v70{};
  pyc::cpp::Wire<32> cache__data__s0__w1{};
  pyc::cpp::Wire<32> cache__data__s1__w0__next{};
  pyc::cpp::Wire<32> v71{};
  pyc::cpp::Wire<32> cache__data__s1__w0{};
  pyc::cpp::Wire<32> cache__data__s1__w1__next{};
  pyc::cpp::Wire<32> v72{};
  pyc::cpp::Wire<32> cache__data__s1__w1{};
  pyc::cpp::Wire<32> cache__data__s2__w0__next{};
  pyc::cpp::Wire<32> v73{};
  pyc::cpp::Wire<32> cache__data__s2__w0{};
  pyc::cpp::Wire<32> cache__data__s2__w1__next{};
  pyc::cpp::Wire<32> v74{};
  pyc::cpp::Wire<32> cache__data__s2__w1{};
  pyc::cpp::Wire<32> cache__data__s3__w0__next{};
  pyc::cpp::Wire<32> v75{};
  pyc::cpp::Wire<32> cache__data__s3__w0{};
  pyc::cpp::Wire<32> cache__data__s3__w1__next{};
  pyc::cpp::Wire<32> v76{};
  pyc::cpp::Wire<32> cache__data__s3__w1{};
  pyc::cpp::Wire<32> cache__data__s4__w0__next{};
  pyc::cpp::Wire<32> v77{};
  pyc::cpp::Wire<32> cache__data__s4__w0{};
  pyc::cpp::Wire<32> cache__data__s4__w1__next{};
  pyc::cpp::Wire<32> v78{};
  pyc::cpp::Wire<32> cache__data__s4__w1{};
  pyc::cpp::Wire<32> cache__data__s5__w0__next{};
  pyc::cpp::Wire<32> v79{};
  pyc::cpp::Wire<32> cache__data__s5__w0{};
  pyc::cpp::Wire<32> cache__data__s5__w1__next{};
  pyc::cpp::Wire<32> v80{};
  pyc::cpp::Wire<32> cache__data__s5__w1{};
  pyc::cpp::Wire<32> cache__data__s6__w0__next{};
  pyc::cpp::Wire<32> v81{};
  pyc::cpp::Wire<32> cache__data__s6__w0{};
  pyc::cpp::Wire<32> cache__data__s6__w1__next{};
  pyc::cpp::Wire<32> v82{};
  pyc::cpp::Wire<32> cache__data__s6__w1{};
  pyc::cpp::Wire<32> cache__data__s7__w0__next{};
  pyc::cpp::Wire<32> v83{};
  pyc::cpp::Wire<32> cache__data__s7__w0{};
  pyc::cpp::Wire<32> cache__data__s7__w1__next{};
  pyc::cpp::Wire<32> v84{};
  pyc::cpp::Wire<32> cache__data__s7__w1{};
  pyc::cpp::Wire<1> cache__rr__s0__next{};
  pyc::cpp::Wire<1> v85{};
  pyc::cpp::Wire<1> cache__rr__s0{};
  pyc::cpp::Wire<1> cache__rr__s1__next{};
  pyc::cpp::Wire<1> v86{};
  pyc::cpp::Wire<1> cache__rr__s1{};
  pyc::cpp::Wire<1> cache__rr__s2__next{};
  pyc::cpp::Wire<1> v87{};
  pyc::cpp::Wire<1> cache__rr__s2{};
  pyc::cpp::Wire<1> cache__rr__s3__next{};
  pyc::cpp::Wire<1> v88{};
  pyc::cpp::Wire<1> cache__rr__s3{};
  pyc::cpp::Wire<1> cache__rr__s4__next{};
  pyc::cpp::Wire<1> v89{};
  pyc::cpp::Wire<1> cache__rr__s4{};
  pyc::cpp::Wire<1> cache__rr__s5__next{};
  pyc::cpp::Wire<1> v90{};
  pyc::cpp::Wire<1> cache__rr__s5{};
  pyc::cpp::Wire<1> cache__rr__s6__next{};
  pyc::cpp::Wire<1> v91{};
  pyc::cpp::Wire<1> cache__rr__s6{};
  pyc::cpp::Wire<1> cache__rr__s7__next{};
  pyc::cpp::Wire<1> v92{};
  pyc::cpp::Wire<1> cache__rr__s7{};
  pyc::cpp::Wire<1> v93{};
  pyc::cpp::Wire<1> cache__req_fire__jit_cache__L194{};
  pyc::cpp::Wire<32> cache__addr__jit_cache__L196{};
  pyc::cpp::Wire<3> v94{};
  pyc::cpp::Wire<3> cache__set_idx__jit_cache__L197{};
  pyc::cpp::Wire<27> v95{};
  pyc::cpp::Wire<27> cache__tag__jit_cache__L198{};
  pyc::cpp::Wire<1> v96{};
  pyc::cpp::Wire<1> v97{};
  pyc::cpp::Wire<1> v98{};
  pyc::cpp::Wire<1> v99{};
  pyc::cpp::Wire<1> v100{};
  pyc::cpp::Wire<1> v101{};
  pyc::cpp::Wire<1> v102{};
  pyc::cpp::Wire<1> v103{};
  pyc::cpp::Wire<1> v104{};
  pyc::cpp::Wire<1> v105{};
  pyc::cpp::Wire<1> v106{};
  pyc::cpp::Wire<1> v107{};
  pyc::cpp::Wire<1> v108{};
  pyc::cpp::Wire<1> v109{};
  pyc::cpp::Wire<1> v110{};
  pyc::cpp::Wire<1> v111{};
  pyc::cpp::Wire<27> v112{};
  pyc::cpp::Wire<27> v113{};
  pyc::cpp::Wire<27> v114{};
  pyc::cpp::Wire<27> v115{};
  pyc::cpp::Wire<27> v116{};
  pyc::cpp::Wire<27> v117{};
  pyc::cpp::Wire<27> v118{};
  pyc::cpp::Wire<27> v119{};
  pyc::cpp::Wire<32> v120{};
  pyc::cpp::Wire<32> v121{};
  pyc::cpp::Wire<32> v122{};
  pyc::cpp::Wire<32> v123{};
  pyc::cpp::Wire<32> v124{};
  pyc::cpp::Wire<32> v125{};
  pyc::cpp::Wire<32> v126{};
  pyc::cpp::Wire<32> v127{};
  pyc::cpp::Wire<1> v128{};
  pyc::cpp::Wire<1> v129{};
  pyc::cpp::Wire<1> v130{};
  pyc::cpp::Wire<32> v131{};
  pyc::cpp::Wire<1> v132{};
  pyc::cpp::Wire<1> v133{};
  pyc::cpp::Wire<1> v134{};
  pyc::cpp::Wire<1> v135{};
  pyc::cpp::Wire<1> v136{};
  pyc::cpp::Wire<1> v137{};
  pyc::cpp::Wire<1> v138{};
  pyc::cpp::Wire<1> v139{};
  pyc::cpp::Wire<27> v140{};
  pyc::cpp::Wire<27> v141{};
  pyc::cpp::Wire<27> v142{};
  pyc::cpp::Wire<27> v143{};
  pyc::cpp::Wire<27> v144{};
  pyc::cpp::Wire<27> v145{};
  pyc::cpp::Wire<27> v146{};
  pyc::cpp::Wire<27> v147{};
  pyc::cpp::Wire<32> v148{};
  pyc::cpp::Wire<32> v149{};
  pyc::cpp::Wire<32> v150{};
  pyc::cpp::Wire<32> v151{};
  pyc::cpp::Wire<32> v152{};
  pyc::cpp::Wire<32> v153{};
  pyc::cpp::Wire<32> v154{};
  pyc::cpp::Wire<32> v155{};
  pyc::cpp::Wire<1> v156{};
  pyc::cpp::Wire<1> v157{};
  pyc::cpp::Wire<1> v158{};
  pyc::cpp::Wire<32> v159{};
  pyc::cpp::Wire<1> v160{};
  pyc::cpp::Wire<1> v161{};
  pyc::cpp::Wire<1> v162{};
  pyc::cpp::Wire<1> v163{};
  pyc::cpp::Wire<1> v164{};
  pyc::cpp::Wire<1> v165{};
  pyc::cpp::Wire<1> v166{};
  pyc::cpp::Wire<1> v167{};
  pyc::cpp::Wire<1> v168{};
  pyc::cpp::Wire<32> v169{};
  pyc::cpp::Wire<1> cache__hit__jit_cache__L202{};
  pyc::cpp::Wire<32> cache__hit_data__jit_cache__L203{};
  pyc::cpp::Wire<32> v170{};
  pyc::cpp::Wire<32> cache__mem_rdata__jit_cache__L206{};
  pyc::cpp::Wire<1> v171{};
  pyc::cpp::Wire<1> v172{};
  pyc::cpp::Wire<1> v173{};
  pyc::cpp::Wire<1> cache__miss__jit_cache__L218{};
  pyc::cpp::Wire<1> v174{};
  pyc::cpp::Wire<1> v175{};
  pyc::cpp::Wire<1> v176{};
  pyc::cpp::Wire<1> v177{};
  pyc::cpp::Wire<1> v178{};
  pyc::cpp::Wire<1> v179{};
  pyc::cpp::Wire<1> v180{};
  pyc::cpp::Wire<1> v181{};
  pyc::cpp::Wire<1> v182{};
  pyc::cpp::Wire<1> cache__repl_way__jit_cache__L219{};
  pyc::cpp::Wire<1> v183{};
  pyc::cpp::Wire<1> v184{};
  pyc::cpp::Wire<1> v185{};
  pyc::cpp::Wire<1> v186{};
  pyc::cpp::Wire<1> v187{};
  pyc::cpp::Wire<1> v188{};
  pyc::cpp::Wire<1> v189{};
  pyc::cpp::Wire<1> v190{};
  pyc::cpp::Wire<1> v191{};
  pyc::cpp::Wire<1> v192{};
  pyc::cpp::Wire<1> v193{};
  pyc::cpp::Wire<1> v194{};
  pyc::cpp::Wire<1> v195{};
  pyc::cpp::Wire<27> v196{};
  pyc::cpp::Wire<32> v197{};
  pyc::cpp::Wire<1> v198{};
  pyc::cpp::Wire<1> v199{};
  pyc::cpp::Wire<1> v200{};
  pyc::cpp::Wire<1> v201{};
  pyc::cpp::Wire<1> v202{};
  pyc::cpp::Wire<1> v203{};
  pyc::cpp::Wire<27> v204{};
  pyc::cpp::Wire<32> v205{};
  pyc::cpp::Wire<1> v206{};
  pyc::cpp::Wire<1> v207{};
  pyc::cpp::Wire<1> v208{};
  pyc::cpp::Wire<1> v209{};
  pyc::cpp::Wire<1> v210{};
  pyc::cpp::Wire<1> v211{};
  pyc::cpp::Wire<1> v212{};
  pyc::cpp::Wire<1> v213{};
  pyc::cpp::Wire<1> v214{};
  pyc::cpp::Wire<1> v215{};
  pyc::cpp::Wire<1> v216{};
  pyc::cpp::Wire<27> v217{};
  pyc::cpp::Wire<32> v218{};
  pyc::cpp::Wire<1> v219{};
  pyc::cpp::Wire<1> v220{};
  pyc::cpp::Wire<1> v221{};
  pyc::cpp::Wire<1> v222{};
  pyc::cpp::Wire<27> v223{};
  pyc::cpp::Wire<32> v224{};
  pyc::cpp::Wire<1> v225{};
  pyc::cpp::Wire<1> v226{};
  pyc::cpp::Wire<1> v227{};
  pyc::cpp::Wire<1> v228{};
  pyc::cpp::Wire<1> v229{};
  pyc::cpp::Wire<1> v230{};
  pyc::cpp::Wire<1> v231{};
  pyc::cpp::Wire<1> v232{};
  pyc::cpp::Wire<1> v233{};
  pyc::cpp::Wire<1> v234{};
  pyc::cpp::Wire<1> v235{};
  pyc::cpp::Wire<27> v236{};
  pyc::cpp::Wire<32> v237{};
  pyc::cpp::Wire<1> v238{};
  pyc::cpp::Wire<1> v239{};
  pyc::cpp::Wire<1> v240{};
  pyc::cpp::Wire<1> v241{};
  pyc::cpp::Wire<27> v242{};
  pyc::cpp::Wire<32> v243{};
  pyc::cpp::Wire<1> v244{};
  pyc::cpp::Wire<1> v245{};
  pyc::cpp::Wire<1> v246{};
  pyc::cpp::Wire<1> v247{};
  pyc::cpp::Wire<1> v248{};
  pyc::cpp::Wire<1> v249{};
  pyc::cpp::Wire<1> v250{};
  pyc::cpp::Wire<1> v251{};
  pyc::cpp::Wire<1> v252{};
  pyc::cpp::Wire<1> v253{};
  pyc::cpp::Wire<1> v254{};
  pyc::cpp::Wire<27> v255{};
  pyc::cpp::Wire<32> v256{};
  pyc::cpp::Wire<1> v257{};
  pyc::cpp::Wire<1> v258{};
  pyc::cpp::Wire<1> v259{};
  pyc::cpp::Wire<1> v260{};
  pyc::cpp::Wire<27> v261{};
  pyc::cpp::Wire<32> v262{};
  pyc::cpp::Wire<1> v263{};
  pyc::cpp::Wire<1> v264{};
  pyc::cpp::Wire<1> v265{};
  pyc::cpp::Wire<1> v266{};
  pyc::cpp::Wire<1> v267{};
  pyc::cpp::Wire<1> v268{};
  pyc::cpp::Wire<1> v269{};
  pyc::cpp::Wire<1> v270{};
  pyc::cpp::Wire<1> v271{};
  pyc::cpp::Wire<1> v272{};
  pyc::cpp::Wire<1> v273{};
  pyc::cpp::Wire<27> v274{};
  pyc::cpp::Wire<32> v275{};
  pyc::cpp::Wire<1> v276{};
  pyc::cpp::Wire<1> v277{};
  pyc::cpp::Wire<1> v278{};
  pyc::cpp::Wire<1> v279{};
  pyc::cpp::Wire<27> v280{};
  pyc::cpp::Wire<32> v281{};
  pyc::cpp::Wire<1> v282{};
  pyc::cpp::Wire<1> v283{};
  pyc::cpp::Wire<1> v284{};
  pyc::cpp::Wire<1> v285{};
  pyc::cpp::Wire<1> v286{};
  pyc::cpp::Wire<1> v287{};
  pyc::cpp::Wire<1> v288{};
  pyc::cpp::Wire<1> v289{};
  pyc::cpp::Wire<1> v290{};
  pyc::cpp::Wire<1> v291{};
  pyc::cpp::Wire<1> v292{};
  pyc::cpp::Wire<27> v293{};
  pyc::cpp::Wire<32> v294{};
  pyc::cpp::Wire<1> v295{};
  pyc::cpp::Wire<1> v296{};
  pyc::cpp::Wire<1> v297{};
  pyc::cpp::Wire<1> v298{};
  pyc::cpp::Wire<27> v299{};
  pyc::cpp::Wire<32> v300{};
  pyc::cpp::Wire<1> v301{};
  pyc::cpp::Wire<1> v302{};
  pyc::cpp::Wire<1> v303{};
  pyc::cpp::Wire<1> v304{};
  pyc::cpp::Wire<1> v305{};
  pyc::cpp::Wire<1> v306{};
  pyc::cpp::Wire<1> v307{};
  pyc::cpp::Wire<1> v308{};
  pyc::cpp::Wire<1> v309{};
  pyc::cpp::Wire<1> v310{};
  pyc::cpp::Wire<1> v311{};
  pyc::cpp::Wire<27> v312{};
  pyc::cpp::Wire<32> v313{};
  pyc::cpp::Wire<1> v314{};
  pyc::cpp::Wire<1> v315{};
  pyc::cpp::Wire<1> v316{};
  pyc::cpp::Wire<1> v317{};
  pyc::cpp::Wire<27> v318{};
  pyc::cpp::Wire<32> v319{};
  pyc::cpp::Wire<1> v320{};
  pyc::cpp::Wire<1> v321{};
  pyc::cpp::Wire<1> v322{};
  pyc::cpp::Wire<1> v323{};
  pyc::cpp::Wire<1> v324{};
  pyc::cpp::Wire<1> v325{};
  pyc::cpp::Wire<1> v326{};
  pyc::cpp::Wire<1> v327{};
  pyc::cpp::Wire<1> v328{};
  pyc::cpp::Wire<1> v329{};
  pyc::cpp::Wire<1> v330{};
  pyc::cpp::Wire<27> v331{};
  pyc::cpp::Wire<32> v332{};
  pyc::cpp::Wire<1> v333{};
  pyc::cpp::Wire<1> v334{};
  pyc::cpp::Wire<1> v335{};
  pyc::cpp::Wire<1> v336{};
  pyc::cpp::Wire<27> v337{};
  pyc::cpp::Wire<32> v338{};
  pyc::cpp::Wire<32> v339{};
  pyc::cpp::Wire<32> cache__rdata__jit_cache__L232{};
  pyc::cpp::Wire<33> v340{};
  pyc::cpp::Wire<33> v341{};
  pyc::cpp::Wire<33> v342{};
  pyc::cpp::Wire<33> v343{};
  pyc::cpp::Wire<33> v344{};
  pyc::cpp::Wire<33> v345{};
  pyc::cpp::Wire<33> cache__rsp_pkt__jit_cache__L233{};

  pyc::cpp::pyc_reg<1> v37_inst;
  pyc::cpp::pyc_reg<1> v38_inst;
  pyc::cpp::pyc_reg<1> v39_inst;
  pyc::cpp::pyc_reg<1> v40_inst;
  pyc::cpp::pyc_reg<1> v41_inst;
  pyc::cpp::pyc_reg<1> v42_inst;
  pyc::cpp::pyc_reg<1> v43_inst;
  pyc::cpp::pyc_reg<1> v44_inst;
  pyc::cpp::pyc_reg<1> v45_inst;
  pyc::cpp::pyc_reg<1> v46_inst;
  pyc::cpp::pyc_reg<1> v47_inst;
  pyc::cpp::pyc_reg<1> v48_inst;
  pyc::cpp::pyc_reg<1> v49_inst;
  pyc::cpp::pyc_reg<1> v50_inst;
  pyc::cpp::pyc_reg<1> v51_inst;
  pyc::cpp::pyc_reg<1> v52_inst;
  pyc::cpp::pyc_reg<27> v53_inst;
  pyc::cpp::pyc_reg<27> v54_inst;
  pyc::cpp::pyc_reg<27> v55_inst;
  pyc::cpp::pyc_reg<27> v56_inst;
  pyc::cpp::pyc_reg<27> v57_inst;
  pyc::cpp::pyc_reg<27> v58_inst;
  pyc::cpp::pyc_reg<27> v59_inst;
  pyc::cpp::pyc_reg<27> v60_inst;
  pyc::cpp::pyc_reg<27> v61_inst;
  pyc::cpp::pyc_reg<27> v62_inst;
  pyc::cpp::pyc_reg<27> v63_inst;
  pyc::cpp::pyc_reg<27> v64_inst;
  pyc::cpp::pyc_reg<27> v65_inst;
  pyc::cpp::pyc_reg<27> v66_inst;
  pyc::cpp::pyc_reg<27> v67_inst;
  pyc::cpp::pyc_reg<27> v68_inst;
  pyc::cpp::pyc_reg<32> v69_inst;
  pyc::cpp::pyc_reg<32> v70_inst;
  pyc::cpp::pyc_reg<32> v71_inst;
  pyc::cpp::pyc_reg<32> v72_inst;
  pyc::cpp::pyc_reg<32> v73_inst;
  pyc::cpp::pyc_reg<32> v74_inst;
  pyc::cpp::pyc_reg<32> v75_inst;
  pyc::cpp::pyc_reg<32> v76_inst;
  pyc::cpp::pyc_reg<32> v77_inst;
  pyc::cpp::pyc_reg<32> v78_inst;
  pyc::cpp::pyc_reg<32> v79_inst;
  pyc::cpp::pyc_reg<32> v80_inst;
  pyc::cpp::pyc_reg<32> v81_inst;
  pyc::cpp::pyc_reg<32> v82_inst;
  pyc::cpp::pyc_reg<32> v83_inst;
  pyc::cpp::pyc_reg<32> v84_inst;
  pyc::cpp::pyc_reg<1> v85_inst;
  pyc::cpp::pyc_reg<1> v86_inst;
  pyc::cpp::pyc_reg<1> v87_inst;
  pyc::cpp::pyc_reg<1> v88_inst;
  pyc::cpp::pyc_reg<1> v89_inst;
  pyc::cpp::pyc_reg<1> v90_inst;
  pyc::cpp::pyc_reg<1> v91_inst;
  pyc::cpp::pyc_reg<1> v92_inst;
  pyc::cpp::pyc_fifo<32, 2> v29_inst;
  pyc::cpp::pyc_fifo<33, 2> v32_inst;
  pyc::cpp::pyc_byte_mem<32, 32, 4096> main_mem;

  JitCache() :
      v37_inst(sys_clk, sys_rst, v25, cache__valid__s0__w0__next, v28, v37),
      v38_inst(sys_clk, sys_rst, v25, cache__valid__s0__w1__next, v28, v38),
      v39_inst(sys_clk, sys_rst, v25, cache__valid__s1__w0__next, v28, v39),
      v40_inst(sys_clk, sys_rst, v25, cache__valid__s1__w1__next, v28, v40),
      v41_inst(sys_clk, sys_rst, v25, cache__valid__s2__w0__next, v28, v41),
      v42_inst(sys_clk, sys_rst, v25, cache__valid__s2__w1__next, v28, v42),
      v43_inst(sys_clk, sys_rst, v25, cache__valid__s3__w0__next, v28, v43),
      v44_inst(sys_clk, sys_rst, v25, cache__valid__s3__w1__next, v28, v44),
      v45_inst(sys_clk, sys_rst, v25, cache__valid__s4__w0__next, v28, v45),
      v46_inst(sys_clk, sys_rst, v25, cache__valid__s4__w1__next, v28, v46),
      v47_inst(sys_clk, sys_rst, v25, cache__valid__s5__w0__next, v28, v47),
      v48_inst(sys_clk, sys_rst, v25, cache__valid__s5__w1__next, v28, v48),
      v49_inst(sys_clk, sys_rst, v25, cache__valid__s6__w0__next, v28, v49),
      v50_inst(sys_clk, sys_rst, v25, cache__valid__s6__w1__next, v28, v50),
      v51_inst(sys_clk, sys_rst, v25, cache__valid__s7__w0__next, v28, v51),
      v52_inst(sys_clk, sys_rst, v25, cache__valid__s7__w1__next, v28, v52),
      v53_inst(sys_clk, sys_rst, v25, cache__tag__s0__w0__next, v24, v53),
      v54_inst(sys_clk, sys_rst, v25, cache__tag__s0__w1__next, v24, v54),
      v55_inst(sys_clk, sys_rst, v25, cache__tag__s1__w0__next, v24, v55),
      v56_inst(sys_clk, sys_rst, v25, cache__tag__s1__w1__next, v24, v56),
      v57_inst(sys_clk, sys_rst, v25, cache__tag__s2__w0__next, v24, v57),
      v58_inst(sys_clk, sys_rst, v25, cache__tag__s2__w1__next, v24, v58),
      v59_inst(sys_clk, sys_rst, v25, cache__tag__s3__w0__next, v24, v59),
      v60_inst(sys_clk, sys_rst, v25, cache__tag__s3__w1__next, v24, v60),
      v61_inst(sys_clk, sys_rst, v25, cache__tag__s4__w0__next, v24, v61),
      v62_inst(sys_clk, sys_rst, v25, cache__tag__s4__w1__next, v24, v62),
      v63_inst(sys_clk, sys_rst, v25, cache__tag__s5__w0__next, v24, v63),
      v64_inst(sys_clk, sys_rst, v25, cache__tag__s5__w1__next, v24, v64),
      v65_inst(sys_clk, sys_rst, v25, cache__tag__s6__w0__next, v24, v65),
      v66_inst(sys_clk, sys_rst, v25, cache__tag__s6__w1__next, v24, v66),
      v67_inst(sys_clk, sys_rst, v25, cache__tag__s7__w0__next, v24, v67),
      v68_inst(sys_clk, sys_rst, v25, cache__tag__s7__w1__next, v24, v68),
      v69_inst(sys_clk, sys_rst, v25, cache__data__s0__w0__next, v27, v69),
      v70_inst(sys_clk, sys_rst, v25, cache__data__s0__w1__next, v27, v70),
      v71_inst(sys_clk, sys_rst, v25, cache__data__s1__w0__next, v27, v71),
      v72_inst(sys_clk, sys_rst, v25, cache__data__s1__w1__next, v27, v72),
      v73_inst(sys_clk, sys_rst, v25, cache__data__s2__w0__next, v27, v73),
      v74_inst(sys_clk, sys_rst, v25, cache__data__s2__w1__next, v27, v74),
      v75_inst(sys_clk, sys_rst, v25, cache__data__s3__w0__next, v27, v75),
      v76_inst(sys_clk, sys_rst, v25, cache__data__s3__w1__next, v27, v76),
      v77_inst(sys_clk, sys_rst, v25, cache__data__s4__w0__next, v27, v77),
      v78_inst(sys_clk, sys_rst, v25, cache__data__s4__w1__next, v27, v78),
      v79_inst(sys_clk, sys_rst, v25, cache__data__s5__w0__next, v27, v79),
      v80_inst(sys_clk, sys_rst, v25, cache__data__s5__w1__next, v27, v80),
      v81_inst(sys_clk, sys_rst, v25, cache__data__s6__w0__next, v27, v81),
      v82_inst(sys_clk, sys_rst, v25, cache__data__s6__w1__next, v27, v82),
      v83_inst(sys_clk, sys_rst, v25, cache__data__s7__w0__next, v27, v83),
      v84_inst(sys_clk, sys_rst, v25, cache__data__s7__w1__next, v27, v84),
      v85_inst(sys_clk, sys_rst, v25, cache__rr__s0__next, v28, v85),
      v86_inst(sys_clk, sys_rst, v25, cache__rr__s1__next, v28, v86),
      v87_inst(sys_clk, sys_rst, v25, cache__rr__s2__next, v28, v87),
      v88_inst(sys_clk, sys_rst, v25, cache__rr__s3__next, v28, v88),
      v89_inst(sys_clk, sys_rst, v25, cache__rr__s4__next, v28, v89),
      v90_inst(sys_clk, sys_rst, v25, cache__rr__s5__next, v28, v90),
      v91_inst(sys_clk, sys_rst, v25, cache__rr__s6__next, v28, v91),
      v92_inst(sys_clk, sys_rst, v25, cache__rr__s7__next, v28, v92),
      v29_inst(sys_clk, sys_rst, cache__req_q__in_valid, v29, cache__req_q__in_data, v30, cache__req_q__out_ready, v31),
      v32_inst(sys_clk, sys_rst, cache__rsp_q__in_valid, v32, cache__rsp_q__in_data, v33, cache__rsp_q__out_ready, v34),
      main_mem(sys_clk, sys_rst, cache__addr__jit_cache__L196, v170, cache__wvalid0__jit_cache__L172, cache__waddr0__jit_cache__L173, cache__wdata0__jit_cache__L174, cache__wstrb0__jit_cache__L175) {
    eval();
  }

  inline void eval_comb_0() {
    v1 = pyc::cpp::Wire<33>(0ull);
    v2 = pyc::cpp::Wire<3>(7ull);
    v3 = pyc::cpp::Wire<3>(6ull);
    v4 = pyc::cpp::Wire<3>(5ull);
    v5 = pyc::cpp::Wire<3>(4ull);
    v6 = pyc::cpp::Wire<3>(3ull);
    v7 = pyc::cpp::Wire<3>(2ull);
    v8 = pyc::cpp::Wire<3>(1ull);
    v9 = pyc::cpp::Wire<3>(0ull);
    v10 = pyc::cpp::Wire<27>(0ull);
    v11 = pyc::cpp::Wire<1>(1ull);
    v12 = pyc::cpp::Wire<4>(0ull);
    v13 = pyc::cpp::Wire<32>(0ull);
    v14 = pyc::cpp::Wire<1>(0ull);
    v15 = v1;
    v16 = v2;
    v17 = v3;
    v18 = v4;
    v19 = v5;
    v20 = v6;
    v21 = v7;
    v22 = v8;
    v23 = v9;
    v24 = v10;
    v25 = v11;
    v26 = v12;
    v27 = v13;
    v28 = v14;
  }

  inline void eval_comb_1() {
    v96 = pyc::cpp::Wire<1>((cache__set_idx__jit_cache__L197 == v23) ? 1u : 0u);
    v97 = (v96.toBool() ? cache__valid__s0__w0 : v28);
    v98 = pyc::cpp::Wire<1>((cache__set_idx__jit_cache__L197 == v22) ? 1u : 0u);
    v99 = (v98.toBool() ? cache__valid__s1__w0 : v97);
    v100 = pyc::cpp::Wire<1>((cache__set_idx__jit_cache__L197 == v21) ? 1u : 0u);
    v101 = (v100.toBool() ? cache__valid__s2__w0 : v99);
    v102 = pyc::cpp::Wire<1>((cache__set_idx__jit_cache__L197 == v20) ? 1u : 0u);
    v103 = (v102.toBool() ? cache__valid__s3__w0 : v101);
    v104 = pyc::cpp::Wire<1>((cache__set_idx__jit_cache__L197 == v19) ? 1u : 0u);
    v105 = (v104.toBool() ? cache__valid__s4__w0 : v103);
    v106 = pyc::cpp::Wire<1>((cache__set_idx__jit_cache__L197 == v18) ? 1u : 0u);
    v107 = (v106.toBool() ? cache__valid__s5__w0 : v105);
    v108 = pyc::cpp::Wire<1>((cache__set_idx__jit_cache__L197 == v17) ? 1u : 0u);
    v109 = (v108.toBool() ? cache__valid__s6__w0 : v107);
    v110 = pyc::cpp::Wire<1>((cache__set_idx__jit_cache__L197 == v16) ? 1u : 0u);
    v111 = (v110.toBool() ? cache__valid__s7__w0 : v109);
    v112 = (v96.toBool() ? cache__tag__s0__w0 : v24);
    v113 = (v98.toBool() ? cache__tag__s1__w0 : v112);
    v114 = (v100.toBool() ? cache__tag__s2__w0 : v113);
    v115 = (v102.toBool() ? cache__tag__s3__w0 : v114);
    v116 = (v104.toBool() ? cache__tag__s4__w0 : v115);
    v117 = (v106.toBool() ? cache__tag__s5__w0 : v116);
    v118 = (v108.toBool() ? cache__tag__s6__w0 : v117);
    v119 = (v110.toBool() ? cache__tag__s7__w0 : v118);
    v120 = (v96.toBool() ? cache__data__s0__w0 : v27);
    v121 = (v98.toBool() ? cache__data__s1__w0 : v120);
    v122 = (v100.toBool() ? cache__data__s2__w0 : v121);
    v123 = (v102.toBool() ? cache__data__s3__w0 : v122);
    v124 = (v104.toBool() ? cache__data__s4__w0 : v123);
    v125 = (v106.toBool() ? cache__data__s5__w0 : v124);
    v126 = (v108.toBool() ? cache__data__s6__w0 : v125);
    v127 = (v110.toBool() ? cache__data__s7__w0 : v126);
    v128 = pyc::cpp::Wire<1>((v119 == cache__tag__jit_cache__L198) ? 1u : 0u);
    v129 = (v111 & v128);
    v130 = (v28 | v129);
    v131 = (v129.toBool() ? v127 : v27);
    v132 = (v96.toBool() ? cache__valid__s0__w1 : v28);
    v133 = (v98.toBool() ? cache__valid__s1__w1 : v132);
    v134 = (v100.toBool() ? cache__valid__s2__w1 : v133);
    v135 = (v102.toBool() ? cache__valid__s3__w1 : v134);
    v136 = (v104.toBool() ? cache__valid__s4__w1 : v135);
    v137 = (v106.toBool() ? cache__valid__s5__w1 : v136);
    v138 = (v108.toBool() ? cache__valid__s6__w1 : v137);
    v139 = (v110.toBool() ? cache__valid__s7__w1 : v138);
    v140 = (v96.toBool() ? cache__tag__s0__w1 : v24);
    v141 = (v98.toBool() ? cache__tag__s1__w1 : v140);
    v142 = (v100.toBool() ? cache__tag__s2__w1 : v141);
    v143 = (v102.toBool() ? cache__tag__s3__w1 : v142);
    v144 = (v104.toBool() ? cache__tag__s4__w1 : v143);
    v145 = (v106.toBool() ? cache__tag__s5__w1 : v144);
    v146 = (v108.toBool() ? cache__tag__s6__w1 : v145);
    v147 = (v110.toBool() ? cache__tag__s7__w1 : v146);
    v148 = (v96.toBool() ? cache__data__s0__w1 : v27);
    v149 = (v98.toBool() ? cache__data__s1__w1 : v148);
    v150 = (v100.toBool() ? cache__data__s2__w1 : v149);
    v151 = (v102.toBool() ? cache__data__s3__w1 : v150);
    v152 = (v104.toBool() ? cache__data__s4__w1 : v151);
    v153 = (v106.toBool() ? cache__data__s5__w1 : v152);
    v154 = (v108.toBool() ? cache__data__s6__w1 : v153);
    v155 = (v110.toBool() ? cache__data__s7__w1 : v154);
    v156 = pyc::cpp::Wire<1>((v147 == cache__tag__jit_cache__L198) ? 1u : 0u);
    v157 = (v139 & v156);
    v158 = (v130 | v157);
    v159 = (v157.toBool() ? v155 : v131);
    v160 = v96;
    v161 = v98;
    v162 = v100;
    v163 = v102;
    v164 = v104;
    v165 = v106;
    v166 = v108;
    v167 = v110;
    v168 = v158;
    v169 = v159;
  }

  inline void eval_comb_2() {
    v171 = (~cache__hit__jit_cache__L202);
    v172 = (cache__req_fire__jit_cache__L194 & v171);
    v173 = v172;
  }

  inline void eval_comb_3() {
    v174 = (v160.toBool() ? cache__rr__s0 : v28);
    v175 = (v161.toBool() ? cache__rr__s1 : v174);
    v176 = (v162.toBool() ? cache__rr__s2 : v175);
    v177 = (v163.toBool() ? cache__rr__s3 : v176);
    v178 = (v164.toBool() ? cache__rr__s4 : v177);
    v179 = (v165.toBool() ? cache__rr__s5 : v178);
    v180 = (v166.toBool() ? cache__rr__s6 : v179);
    v181 = (v167.toBool() ? cache__rr__s7 : v180);
    v182 = v181;
  }

  inline void eval_comb_4() {
    v183 = (cache__miss__jit_cache__L218 & v160);
    v184 = (cache__rr__s0 + v25);
    v185 = pyc::cpp::Wire<1>((cache__rr__s0 == v25) ? 1u : 0u);
    v186 = (v185.toBool() ? v28 : v184);
    v187 = (v183.toBool() ? v186 : cache__rr__s0);
    v188 = v183;
    v189 = v187;
  }

  inline void eval_comb_5() {
    v190 = pyc::cpp::Wire<1>((cache__repl_way__jit_cache__L219 == v28) ? 1u : 0u);
    v191 = (v188 & v190);
    v192 = (v191.toBool() ? v25 : cache__valid__s0__w0);
    v193 = v190;
    v194 = v191;
    v195 = v192;
  }

  inline void eval_comb_6() {
    v198 = pyc::cpp::Wire<1>((cache__repl_way__jit_cache__L219 == v25) ? 1u : 0u);
    v199 = (v188 & v198);
    v200 = (v199.toBool() ? v25 : cache__valid__s0__w1);
    v201 = v198;
    v202 = v199;
    v203 = v200;
  }

  inline void eval_comb_7() {
    v206 = (cache__miss__jit_cache__L218 & v161);
    v207 = (cache__rr__s1 + v25);
    v208 = pyc::cpp::Wire<1>((cache__rr__s1 == v25) ? 1u : 0u);
    v209 = (v208.toBool() ? v28 : v207);
    v210 = (v206.toBool() ? v209 : cache__rr__s1);
    v211 = v206;
    v212 = v210;
  }

  inline void eval_comb_8() {
    v213 = (v211 & v193);
    v214 = (v213.toBool() ? v25 : cache__valid__s1__w0);
    v215 = v213;
    v216 = v214;
  }

  inline void eval_comb_9() {
    v219 = (v211 & v201);
    v220 = (v219.toBool() ? v25 : cache__valid__s1__w1);
    v221 = v219;
    v222 = v220;
  }

  inline void eval_comb_10() {
    v225 = (cache__miss__jit_cache__L218 & v162);
    v226 = (cache__rr__s2 + v25);
    v227 = pyc::cpp::Wire<1>((cache__rr__s2 == v25) ? 1u : 0u);
    v228 = (v227.toBool() ? v28 : v226);
    v229 = (v225.toBool() ? v228 : cache__rr__s2);
    v230 = v225;
    v231 = v229;
  }

  inline void eval_comb_11() {
    v232 = (v230 & v193);
    v233 = (v232.toBool() ? v25 : cache__valid__s2__w0);
    v234 = v232;
    v235 = v233;
  }

  inline void eval_comb_12() {
    v238 = (v230 & v201);
    v239 = (v238.toBool() ? v25 : cache__valid__s2__w1);
    v240 = v238;
    v241 = v239;
  }

  inline void eval_comb_13() {
    v244 = (cache__miss__jit_cache__L218 & v163);
    v245 = (cache__rr__s3 + v25);
    v246 = pyc::cpp::Wire<1>((cache__rr__s3 == v25) ? 1u : 0u);
    v247 = (v246.toBool() ? v28 : v245);
    v248 = (v244.toBool() ? v247 : cache__rr__s3);
    v249 = v244;
    v250 = v248;
  }

  inline void eval_comb_14() {
    v251 = (v249 & v193);
    v252 = (v251.toBool() ? v25 : cache__valid__s3__w0);
    v253 = v251;
    v254 = v252;
  }

  inline void eval_comb_15() {
    v257 = (v249 & v201);
    v258 = (v257.toBool() ? v25 : cache__valid__s3__w1);
    v259 = v257;
    v260 = v258;
  }

  inline void eval_comb_16() {
    v263 = (cache__miss__jit_cache__L218 & v164);
    v264 = (cache__rr__s4 + v25);
    v265 = pyc::cpp::Wire<1>((cache__rr__s4 == v25) ? 1u : 0u);
    v266 = (v265.toBool() ? v28 : v264);
    v267 = (v263.toBool() ? v266 : cache__rr__s4);
    v268 = v263;
    v269 = v267;
  }

  inline void eval_comb_17() {
    v270 = (v268 & v193);
    v271 = (v270.toBool() ? v25 : cache__valid__s4__w0);
    v272 = v270;
    v273 = v271;
  }

  inline void eval_comb_18() {
    v276 = (v268 & v201);
    v277 = (v276.toBool() ? v25 : cache__valid__s4__w1);
    v278 = v276;
    v279 = v277;
  }

  inline void eval_comb_19() {
    v282 = (cache__miss__jit_cache__L218 & v165);
    v283 = (cache__rr__s5 + v25);
    v284 = pyc::cpp::Wire<1>((cache__rr__s5 == v25) ? 1u : 0u);
    v285 = (v284.toBool() ? v28 : v283);
    v286 = (v282.toBool() ? v285 : cache__rr__s5);
    v287 = v282;
    v288 = v286;
  }

  inline void eval_comb_20() {
    v289 = (v287 & v193);
    v290 = (v289.toBool() ? v25 : cache__valid__s5__w0);
    v291 = v289;
    v292 = v290;
  }

  inline void eval_comb_21() {
    v295 = (v287 & v201);
    v296 = (v295.toBool() ? v25 : cache__valid__s5__w1);
    v297 = v295;
    v298 = v296;
  }

  inline void eval_comb_22() {
    v301 = (cache__miss__jit_cache__L218 & v166);
    v302 = (cache__rr__s6 + v25);
    v303 = pyc::cpp::Wire<1>((cache__rr__s6 == v25) ? 1u : 0u);
    v304 = (v303.toBool() ? v28 : v302);
    v305 = (v301.toBool() ? v304 : cache__rr__s6);
    v306 = v301;
    v307 = v305;
  }

  inline void eval_comb_23() {
    v308 = (v306 & v193);
    v309 = (v308.toBool() ? v25 : cache__valid__s6__w0);
    v310 = v308;
    v311 = v309;
  }

  inline void eval_comb_24() {
    v314 = (v306 & v201);
    v315 = (v314.toBool() ? v25 : cache__valid__s6__w1);
    v316 = v314;
    v317 = v315;
  }

  inline void eval_comb_25() {
    v320 = (cache__miss__jit_cache__L218 & v167);
    v321 = (cache__rr__s7 + v25);
    v322 = pyc::cpp::Wire<1>((cache__rr__s7 == v25) ? 1u : 0u);
    v323 = (v322.toBool() ? v28 : v321);
    v324 = (v320.toBool() ? v323 : cache__rr__s7);
    v325 = v320;
    v326 = v324;
  }

  inline void eval_comb_26() {
    v327 = (v325 & v193);
    v328 = (v327.toBool() ? v25 : cache__valid__s7__w0);
    v329 = v327;
    v330 = v328;
  }

  inline void eval_comb_27() {
    v333 = (v325 & v201);
    v334 = (v333.toBool() ? v25 : cache__valid__s7__w1);
    v335 = v333;
    v336 = v334;
  }

  inline void eval_comb_28() {
    v340 = pyc::cpp::zext<33, 32>(cache__rdata__jit_cache__L232);
    v341 = (v15 | v340);
    v342 = pyc::cpp::zext<33, 1>(cache__hit__jit_cache__L202);
    v343 = pyc::cpp::Wire<33>(v342.value() << 32ull);
    v344 = (v341 | v343);
    v345 = v344;
  }

  inline void eval_comb_pass() {
    eval_comb_0();
    req_valid__jit_cache__L166 = req_valid;
    req_addr__jit_cache__L167 = req_addr;
    rsp_ready__jit_cache__L168 = rsp_ready;
    cache__wvalid0__jit_cache__L172 = v28;
    cache__waddr0__jit_cache__L173 = v27;
    cache__wdata0__jit_cache__L174 = v27;
    cache__wstrb0__jit_cache__L175 = v26;
    v35 = pyc::cpp::extract<1, 33>(v34, 32u);
    cache__rsp_hit__jit_cache__L183 = v35;
    v36 = pyc::cpp::extract<32, 33>(v34, 0u);
    cache__rsp_rdata__jit_cache__L184 = v36;
    cache__valid__s0__w0 = v37;
    cache__valid__s0__w1 = v38;
    cache__valid__s1__w0 = v39;
    cache__valid__s1__w1 = v40;
    cache__valid__s2__w0 = v41;
    cache__valid__s2__w1 = v42;
    cache__valid__s3__w0 = v43;
    cache__valid__s3__w1 = v44;
    cache__valid__s4__w0 = v45;
    cache__valid__s4__w1 = v46;
    cache__valid__s5__w0 = v47;
    cache__valid__s5__w1 = v48;
    cache__valid__s6__w0 = v49;
    cache__valid__s6__w1 = v50;
    cache__valid__s7__w0 = v51;
    cache__valid__s7__w1 = v52;
    cache__tag__s0__w0 = v53;
    cache__tag__s0__w1 = v54;
    cache__tag__s1__w0 = v55;
    cache__tag__s1__w1 = v56;
    cache__tag__s2__w0 = v57;
    cache__tag__s2__w1 = v58;
    cache__tag__s3__w0 = v59;
    cache__tag__s3__w1 = v60;
    cache__tag__s4__w0 = v61;
    cache__tag__s4__w1 = v62;
    cache__tag__s5__w0 = v63;
    cache__tag__s5__w1 = v64;
    cache__tag__s6__w0 = v65;
    cache__tag__s6__w1 = v66;
    cache__tag__s7__w0 = v67;
    cache__tag__s7__w1 = v68;
    cache__data__s0__w0 = v69;
    cache__data__s0__w1 = v70;
    cache__data__s1__w0 = v71;
    cache__data__s1__w1 = v72;
    cache__data__s2__w0 = v73;
    cache__data__s2__w1 = v74;
    cache__data__s3__w0 = v75;
    cache__data__s3__w1 = v76;
    cache__data__s4__w0 = v77;
    cache__data__s4__w1 = v78;
    cache__data__s5__w0 = v79;
    cache__data__s5__w1 = v80;
    cache__data__s6__w0 = v81;
    cache__data__s6__w1 = v82;
    cache__data__s7__w0 = v83;
    cache__data__s7__w1 = v84;
    cache__rr__s0 = v85;
    cache__rr__s1 = v86;
    cache__rr__s2 = v87;
    cache__rr__s3 = v88;
    cache__rr__s4 = v89;
    cache__rr__s5 = v90;
    cache__rr__s6 = v91;
    cache__rr__s7 = v92;
    v93 = (v30 & v32);
    cache__req_fire__jit_cache__L194 = v93;
    cache__addr__jit_cache__L196 = v31;
    v94 = pyc::cpp::extract<3, 32>(cache__addr__jit_cache__L196, 2u);
    cache__set_idx__jit_cache__L197 = v94;
    v95 = pyc::cpp::extract<27, 32>(cache__addr__jit_cache__L196, 5u);
    cache__tag__jit_cache__L198 = v95;
    eval_comb_1();
    cache__hit__jit_cache__L202 = v168;
    cache__hit_data__jit_cache__L203 = v169;
    cache__mem_rdata__jit_cache__L206 = v170;
    eval_comb_2();
    cache__miss__jit_cache__L218 = v173;
    eval_comb_3();
    cache__repl_way__jit_cache__L219 = v182;
    eval_comb_4();
    cache__rr__s0__next = v189;
    eval_comb_5();
    cache__valid__s0__w0__next = v195;
    v196 = (v194.toBool() ? cache__tag__jit_cache__L198 : cache__tag__s0__w0);
    cache__tag__s0__w0__next = v196;
    v197 = (v194.toBool() ? cache__mem_rdata__jit_cache__L206 : cache__data__s0__w0);
    cache__data__s0__w0__next = v197;
    eval_comb_6();
    cache__valid__s0__w1__next = v203;
    v204 = (v202.toBool() ? cache__tag__jit_cache__L198 : cache__tag__s0__w1);
    cache__tag__s0__w1__next = v204;
    v205 = (v202.toBool() ? cache__mem_rdata__jit_cache__L206 : cache__data__s0__w1);
    cache__data__s0__w1__next = v205;
    eval_comb_7();
    cache__rr__s1__next = v212;
    eval_comb_8();
    cache__valid__s1__w0__next = v216;
    v217 = (v215.toBool() ? cache__tag__jit_cache__L198 : cache__tag__s1__w0);
    cache__tag__s1__w0__next = v217;
    v218 = (v215.toBool() ? cache__mem_rdata__jit_cache__L206 : cache__data__s1__w0);
    cache__data__s1__w0__next = v218;
    eval_comb_9();
    cache__valid__s1__w1__next = v222;
    v223 = (v221.toBool() ? cache__tag__jit_cache__L198 : cache__tag__s1__w1);
    cache__tag__s1__w1__next = v223;
    v224 = (v221.toBool() ? cache__mem_rdata__jit_cache__L206 : cache__data__s1__w1);
    cache__data__s1__w1__next = v224;
    eval_comb_10();
    cache__rr__s2__next = v231;
    eval_comb_11();
    cache__valid__s2__w0__next = v235;
    v236 = (v234.toBool() ? cache__tag__jit_cache__L198 : cache__tag__s2__w0);
    cache__tag__s2__w0__next = v236;
    v237 = (v234.toBool() ? cache__mem_rdata__jit_cache__L206 : cache__data__s2__w0);
    cache__data__s2__w0__next = v237;
    eval_comb_12();
    cache__valid__s2__w1__next = v241;
    v242 = (v240.toBool() ? cache__tag__jit_cache__L198 : cache__tag__s2__w1);
    cache__tag__s2__w1__next = v242;
    v243 = (v240.toBool() ? cache__mem_rdata__jit_cache__L206 : cache__data__s2__w1);
    cache__data__s2__w1__next = v243;
    eval_comb_13();
    cache__rr__s3__next = v250;
    eval_comb_14();
    cache__valid__s3__w0__next = v254;
    v255 = (v253.toBool() ? cache__tag__jit_cache__L198 : cache__tag__s3__w0);
    cache__tag__s3__w0__next = v255;
    v256 = (v253.toBool() ? cache__mem_rdata__jit_cache__L206 : cache__data__s3__w0);
    cache__data__s3__w0__next = v256;
    eval_comb_15();
    cache__valid__s3__w1__next = v260;
    v261 = (v259.toBool() ? cache__tag__jit_cache__L198 : cache__tag__s3__w1);
    cache__tag__s3__w1__next = v261;
    v262 = (v259.toBool() ? cache__mem_rdata__jit_cache__L206 : cache__data__s3__w1);
    cache__data__s3__w1__next = v262;
    eval_comb_16();
    cache__rr__s4__next = v269;
    eval_comb_17();
    cache__valid__s4__w0__next = v273;
    v274 = (v272.toBool() ? cache__tag__jit_cache__L198 : cache__tag__s4__w0);
    cache__tag__s4__w0__next = v274;
    v275 = (v272.toBool() ? cache__mem_rdata__jit_cache__L206 : cache__data__s4__w0);
    cache__data__s4__w0__next = v275;
    eval_comb_18();
    cache__valid__s4__w1__next = v279;
    v280 = (v278.toBool() ? cache__tag__jit_cache__L198 : cache__tag__s4__w1);
    cache__tag__s4__w1__next = v280;
    v281 = (v278.toBool() ? cache__mem_rdata__jit_cache__L206 : cache__data__s4__w1);
    cache__data__s4__w1__next = v281;
    eval_comb_19();
    cache__rr__s5__next = v288;
    eval_comb_20();
    cache__valid__s5__w0__next = v292;
    v293 = (v291.toBool() ? cache__tag__jit_cache__L198 : cache__tag__s5__w0);
    cache__tag__s5__w0__next = v293;
    v294 = (v291.toBool() ? cache__mem_rdata__jit_cache__L206 : cache__data__s5__w0);
    cache__data__s5__w0__next = v294;
    eval_comb_21();
    cache__valid__s5__w1__next = v298;
    v299 = (v297.toBool() ? cache__tag__jit_cache__L198 : cache__tag__s5__w1);
    cache__tag__s5__w1__next = v299;
    v300 = (v297.toBool() ? cache__mem_rdata__jit_cache__L206 : cache__data__s5__w1);
    cache__data__s5__w1__next = v300;
    eval_comb_22();
    cache__rr__s6__next = v307;
    eval_comb_23();
    cache__valid__s6__w0__next = v311;
    v312 = (v310.toBool() ? cache__tag__jit_cache__L198 : cache__tag__s6__w0);
    cache__tag__s6__w0__next = v312;
    v313 = (v310.toBool() ? cache__mem_rdata__jit_cache__L206 : cache__data__s6__w0);
    cache__data__s6__w0__next = v313;
    eval_comb_24();
    cache__valid__s6__w1__next = v317;
    v318 = (v316.toBool() ? cache__tag__jit_cache__L198 : cache__tag__s6__w1);
    cache__tag__s6__w1__next = v318;
    v319 = (v316.toBool() ? cache__mem_rdata__jit_cache__L206 : cache__data__s6__w1);
    cache__data__s6__w1__next = v319;
    eval_comb_25();
    cache__rr__s7__next = v326;
    eval_comb_26();
    cache__valid__s7__w0__next = v330;
    v331 = (v329.toBool() ? cache__tag__jit_cache__L198 : cache__tag__s7__w0);
    cache__tag__s7__w0__next = v331;
    v332 = (v329.toBool() ? cache__mem_rdata__jit_cache__L206 : cache__data__s7__w0);
    cache__data__s7__w0__next = v332;
    eval_comb_27();
    cache__valid__s7__w1__next = v336;
    v337 = (v335.toBool() ? cache__tag__jit_cache__L198 : cache__tag__s7__w1);
    cache__tag__s7__w1__next = v337;
    v338 = (v335.toBool() ? cache__mem_rdata__jit_cache__L206 : cache__data__s7__w1);
    cache__data__s7__w1__next = v338;
    v339 = (cache__hit__jit_cache__L202.toBool() ? cache__hit_data__jit_cache__L203 : cache__mem_rdata__jit_cache__L206);
    cache__rdata__jit_cache__L232 = v339;
    eval_comb_28();
    cache__rsp_pkt__jit_cache__L233 = v345;
    cache__req_q__in_valid = req_valid__jit_cache__L166;
    cache__req_q__in_data = req_addr__jit_cache__L167;
    cache__req_q__out_ready = v32;
    cache__rsp_q__in_valid = cache__req_fire__jit_cache__L194;
    cache__rsp_q__in_data = cache__rsp_pkt__jit_cache__L233;
    cache__rsp_q__out_ready = rsp_ready__jit_cache__L168;
  }

  void eval() {
    eval_comb_pass();
    for (unsigned _i = 0; _i < 3u; ++_i) {
      v29_inst.eval();
      v32_inst.eval();
      main_mem.eval();
      eval_comb_pass();
    }
    req_ready = v29;
    rsp_valid = v33;
    rsp_hit = cache__rsp_hit__jit_cache__L183;
    rsp_rdata = cache__rsp_rdata__jit_cache__L184;
  }

  void tick() {
    // Two-phase update: compute next state for all sequential elements,
    // then commit together. This avoids ordering artifacts between regs.
    // Phase 1: compute.
    v29_inst.tick_compute();
    v32_inst.tick_compute();
    v37_inst.tick_compute();
    v38_inst.tick_compute();
    v39_inst.tick_compute();
    v40_inst.tick_compute();
    v41_inst.tick_compute();
    v42_inst.tick_compute();
    v43_inst.tick_compute();
    v44_inst.tick_compute();
    v45_inst.tick_compute();
    v46_inst.tick_compute();
    v47_inst.tick_compute();
    v48_inst.tick_compute();
    v49_inst.tick_compute();
    v50_inst.tick_compute();
    v51_inst.tick_compute();
    v52_inst.tick_compute();
    v53_inst.tick_compute();
    v54_inst.tick_compute();
    v55_inst.tick_compute();
    v56_inst.tick_compute();
    v57_inst.tick_compute();
    v58_inst.tick_compute();
    v59_inst.tick_compute();
    v60_inst.tick_compute();
    v61_inst.tick_compute();
    v62_inst.tick_compute();
    v63_inst.tick_compute();
    v64_inst.tick_compute();
    v65_inst.tick_compute();
    v66_inst.tick_compute();
    v67_inst.tick_compute();
    v68_inst.tick_compute();
    v69_inst.tick_compute();
    v70_inst.tick_compute();
    v71_inst.tick_compute();
    v72_inst.tick_compute();
    v73_inst.tick_compute();
    v74_inst.tick_compute();
    v75_inst.tick_compute();
    v76_inst.tick_compute();
    v77_inst.tick_compute();
    v78_inst.tick_compute();
    v79_inst.tick_compute();
    v80_inst.tick_compute();
    v81_inst.tick_compute();
    v82_inst.tick_compute();
    v83_inst.tick_compute();
    v84_inst.tick_compute();
    v85_inst.tick_compute();
    v86_inst.tick_compute();
    v87_inst.tick_compute();
    v88_inst.tick_compute();
    v89_inst.tick_compute();
    v90_inst.tick_compute();
    v91_inst.tick_compute();
    v92_inst.tick_compute();
    main_mem.tick_compute();
    // Phase 2: commit.
    v29_inst.tick_commit();
    v32_inst.tick_commit();
    v37_inst.tick_commit();
    v38_inst.tick_commit();
    v39_inst.tick_commit();
    v40_inst.tick_commit();
    v41_inst.tick_commit();
    v42_inst.tick_commit();
    v43_inst.tick_commit();
    v44_inst.tick_commit();
    v45_inst.tick_commit();
    v46_inst.tick_commit();
    v47_inst.tick_commit();
    v48_inst.tick_commit();
    v49_inst.tick_commit();
    v50_inst.tick_commit();
    v51_inst.tick_commit();
    v52_inst.tick_commit();
    v53_inst.tick_commit();
    v54_inst.tick_commit();
    v55_inst.tick_commit();
    v56_inst.tick_commit();
    v57_inst.tick_commit();
    v58_inst.tick_commit();
    v59_inst.tick_commit();
    v60_inst.tick_commit();
    v61_inst.tick_commit();
    v62_inst.tick_commit();
    v63_inst.tick_commit();
    v64_inst.tick_commit();
    v65_inst.tick_commit();
    v66_inst.tick_commit();
    v67_inst.tick_commit();
    v68_inst.tick_commit();
    v69_inst.tick_commit();
    v70_inst.tick_commit();
    v71_inst.tick_commit();
    v72_inst.tick_commit();
    v73_inst.tick_commit();
    v74_inst.tick_commit();
    v75_inst.tick_commit();
    v76_inst.tick_commit();
    v77_inst.tick_commit();
    v78_inst.tick_commit();
    v79_inst.tick_commit();
    v80_inst.tick_commit();
    v81_inst.tick_commit();
    v82_inst.tick_commit();
    v83_inst.tick_commit();
    v84_inst.tick_commit();
    v85_inst.tick_commit();
    v86_inst.tick_commit();
    v87_inst.tick_commit();
    v88_inst.tick_commit();
    v89_inst.tick_commit();
    v90_inst.tick_commit();
    v91_inst.tick_commit();
    v92_inst.tick_commit();
    main_mem.tick_commit();
  }
};

} // namespace pyc::gen
