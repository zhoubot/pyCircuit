from __future__ import annotations

from pycircuit import CycleAwareCircuit, CycleAwareSignal, mux

from ..isa import (
    BK_CALL,
    BK_COND,
    BK_FALL,
    BK_RET,
    OP_BSTART_STD_CALL,
    OP_C_BSTART_COND,
    OP_C_BSTART_STD,
    OP_C_LWI,
    OP_C_SETC_EQ,
    OP_C_SETC_TGT,
    OP_C_SWI,
    OP_C_BSTOP,
    OP_SWI,
    REG_INVALID,
    ST_EX,
    ST_ID,
    ST_IF,
    ST_MEM,
    ST_WB,
)
from ..pipeline import CoreState, MemWbRegs, RegFiles
from ..regfile import commit_gpr, commit_stack, stack_next


def build_wb_stage(
    m: CycleAwareCircuit,
    *,
    do_wb: CycleAwareSignal,
    stage_is_if: CycleAwareSignal,
    stage_is_id: CycleAwareSignal,
    stage_is_ex: CycleAwareSignal,
    stage_is_mem: CycleAwareSignal,
    stage_is_wb: CycleAwareSignal,
    stop: CycleAwareSignal,
    halt_set: CycleAwareSignal,
    state: CoreState,
    memwb: MemWbRegs,
    rf: RegFiles,
) -> None:
    c = m.ca_const
    
    # Stage inputs.
    stage = state.stage.out()
    pc = state.pc.out()
    br_kind = state.br_kind.out()
    br_base_pc = state.br_base_pc.out()
    br_off = state.br_off.out()
    commit_cond = state.commit_cond.out()
    commit_tgt = state.commit_tgt.out()

    op = memwb.op.out()
    len_bytes = memwb.len_bytes.out()
    regdst = memwb.regdst.out()
    value = memwb.value.out()

    # Halt flag (latches even when stop inhibits do_wb).
    state.halted.set(c(1, width=1), when=halt_set)

    # --- BlockISA control flow ---
    op_c_bstart_std = op.eq(OP_C_BSTART_STD)
    op_c_bstart_cond = op.eq(OP_C_BSTART_COND)
    op_bstart_call = op.eq(OP_BSTART_STD_CALL)
    op_c_bstop = op.eq(OP_C_BSTOP)

    op_is_start_marker = op_c_bstart_std | op_c_bstart_cond | op_bstart_call
    op_is_boundary = op_is_start_marker | op_c_bstop

    br_is_cond = br_kind.eq(BK_COND)
    br_is_call = br_kind.eq(BK_CALL)
    br_is_ret = br_kind.eq(BK_RET)

    br_target_pc_base = br_base_pc + br_off
    br_target_pc = mux(br_is_ret, commit_tgt, br_target_pc_base)

    br_take = br_is_call | br_is_ret | (br_is_cond & commit_cond)

    pc_inc = pc + len_bytes
    pc_next = mux(op_is_boundary & br_take, br_target_pc, pc_inc)
    state.pc.set(pc_next, when=do_wb)

    # Stage machine: IF -> ID -> EX -> MEM -> WB -> IF
    stage_seq = stage
    stage_seq = mux(stage_is_if, c(ST_ID, width=3), stage_seq)
    stage_seq = mux(stage_is_id, c(ST_EX, width=3), stage_seq)
    stage_seq = mux(stage_is_ex, c(ST_MEM, width=3), stage_seq)
    stage_seq = mux(stage_is_mem, c(ST_WB, width=3), stage_seq)
    stage_seq = mux(stage_is_wb, c(ST_IF, width=3), stage_seq)
    state.stage.set(stage_seq, when=~stop)

    # Cycle counter
    state.cycles.set(state.cycles.out() + 1)

    # --- Block control state updates ---
    op_c_setc_eq = op.eq(OP_C_SETC_EQ)
    op_c_setc_tgt = op.eq(OP_C_SETC_TGT)

    # Clear commit args at boundary markers
    commit_cond_cleared = mux(do_wb & op_is_boundary, c(0, width=1), commit_cond)
    commit_tgt_cleared = mux(do_wb & op_is_boundary, c(0, width=64), commit_tgt)
    
    # Update on SETC.EQ / SETC.TGT
    commit_cond_next = mux(do_wb & op_c_setc_eq, value[0], commit_cond_cleared)
    commit_tgt_next = mux(do_wb & op_c_setc_tgt, value, commit_tgt_cleared)
    
    state.commit_cond.set(commit_cond_next)
    state.commit_tgt.set(commit_tgt_next)

    # Block transition kind
    br_kind_base = br_kind
    br_base_base = br_base_pc
    br_off_base = br_off

    # Reset when leaving block via boundary
    leave_block = do_wb & op_is_boundary & br_take
    br_kind_base = mux(leave_block, c(BK_FALL, width=2), br_kind_base)
    br_base_base = mux(leave_block, pc, br_base_base)
    br_off_base = mux(leave_block, c(0, width=64), br_off_base)

    enter_new_block = do_wb & op_is_start_marker & (~br_take)

    # C.BSTART COND
    br_kind_next = mux(enter_new_block & op_c_bstart_cond, c(BK_COND, width=2), br_kind_base)
    br_base_next = mux(enter_new_block & op_c_bstart_cond, pc, br_base_base)
    br_off_next = mux(enter_new_block & op_c_bstart_cond, value, br_off_base)

    # BSTART.STD CALL
    br_kind_next = mux(enter_new_block & op_bstart_call, c(BK_CALL, width=2), br_kind_next)
    br_base_next = mux(enter_new_block & op_bstart_call, pc, br_base_next)
    br_off_next = mux(enter_new_block & op_bstart_call, value, br_off_next)

    # C.BSTART.STD BrType
    brtype = value[0:3]
    kind_from_brtype = mux(brtype.eq(7), c(BK_RET, width=2), c(BK_FALL, width=2))
    br_kind_next = mux(enter_new_block & op_c_bstart_std, kind_from_brtype, br_kind_next)
    br_base_next = mux(enter_new_block & op_c_bstart_std, pc, br_base_next)
    br_off_next = mux(enter_new_block & op_c_bstart_std, c(0, width=64), br_off_next)

    # C.BSTOP
    br_kind_next = mux(do_wb & op_c_bstop, c(BK_FALL, width=2), br_kind_next)
    br_base_next = mux(do_wb & op_c_bstop, pc, br_base_next)
    br_off_next = mux(do_wb & op_c_bstop, c(0, width=64), br_off_next)

    state.br_kind.set(br_kind_next)
    state.br_base_pc.set(br_base_next)
    state.br_off.set(br_off_next)

    # Register writeback + T/U stacks
    wb_is_store = op.eq(OP_SWI) | op.eq(OP_C_SWI)
    do_reg_write = do_wb & (~wb_is_store) & regdst.ne(REG_INVALID)

    do_clear_hands = do_wb & op_is_start_marker
    do_push_t = do_wb & op.eq(OP_C_LWI)
    do_push_t = do_push_t | (do_reg_write & regdst.eq(31))
    do_push_u = do_reg_write & regdst.eq(30)

    commit_gpr(m, rf.gpr, do_reg_write=do_reg_write, regdst=regdst, value=value)

    t_next = stack_next(m, rf.t, do_push=do_push_t, do_clear=do_clear_hands, value=value)
    u_next = stack_next(m, rf.u, do_push=do_push_u, do_clear=do_clear_hands, value=value)
    commit_stack(m, rf.t, t_next)
    commit_stack(m, rf.u, u_next)
