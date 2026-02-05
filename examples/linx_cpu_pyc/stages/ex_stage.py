from __future__ import annotations

from pycircuit import CycleAwareCircuit, CycleAwareSignal, mux

from ..isa import (
    OP_ADDTPC,
    OP_ADDI,
    OP_ADDIW,
    OP_ADDW,
    OP_ANDW,
    OP_BSTART_STD_CALL,
    OP_CMP_EQ,
    OP_C_BSTART_COND,
    OP_C_BSTART_STD,
    OP_CSEL,
    OP_C_LWI,
    OP_C_MOVI,
    OP_C_MOVR,
    OP_C_SETC_EQ,
    OP_C_SETC_TGT,
    OP_C_SETRET,
    OP_C_SWI,
    OP_HL_LUI,
    OP_LWI,
    OP_ORW,
    OP_SDI,
    OP_SUBI,
    OP_SWI,
    OP_XORW,
)
from ..pipeline import ExMemRegs, IdExRegs
from ..util import Consts


def build_ex_stage(
    m: CycleAwareCircuit,
    *,
    do_ex: CycleAwareSignal,
    pc: CycleAwareSignal,
    idex: IdExRegs,
    exmem: ExMemRegs,
    consts: Consts,
) -> None:
    z1 = consts.zero1
    z3 = consts.zero3
    z64 = consts.zero64
    one64 = consts.one64

    # Stage inputs.
    pc_val = pc
    op = idex.op.out()
    len_bytes = idex.len_bytes.out()
    regdst = idex.regdst.out()
    srcl_val = idex.srcl_val.out()
    srcr_val = idex.srcr_val.out()
    srcp_val = idex.srcp_val.out()
    imm = idex.imm.out()

    op_c_bstart_std = op.eq(OP_C_BSTART_STD)
    op_c_bstart_cond = op.eq(OP_C_BSTART_COND)
    op_bstart_std_call = op.eq(OP_BSTART_STD_CALL)
    op_c_movr = op.eq(OP_C_MOVR)
    op_c_movi = op.eq(OP_C_MOVI)
    op_c_setret = op.eq(OP_C_SETRET)
    op_c_setc_eq = op.eq(OP_C_SETC_EQ)
    op_c_setc_tgt = op.eq(OP_C_SETC_TGT)
    op_addtpc = op.eq(OP_ADDTPC)
    op_addi = op.eq(OP_ADDI)
    op_subi = op.eq(OP_SUBI)
    op_addiw = op.eq(OP_ADDIW)
    op_addw = op.eq(OP_ADDW)
    op_orw = op.eq(OP_ORW)
    op_andw = op.eq(OP_ANDW)
    op_xorw = op.eq(OP_XORW)
    op_cmp_eq = op.eq(OP_CMP_EQ)
    op_csel = op.eq(OP_CSEL)
    op_hl_lui = op.eq(OP_HL_LUI)
    op_lwi = op.eq(OP_LWI)
    op_c_lwi = op.eq(OP_C_LWI)
    op_swi = op.eq(OP_SWI)
    op_c_swi = op.eq(OP_C_SWI)
    op_sdi = op.eq(OP_SDI)

    off = imm.shl(amount=2)

    # Default values
    alu = z64
    is_load = z1
    is_store = z1
    size = z3
    addr = z64
    wdata = z64

    # Block markers: forward imm through ALU
    is_block_marker = op_c_bstart_std | op_c_bstart_cond | op_bstart_std_call
    alu = mux(is_block_marker, imm, alu)

    # MOVR: pass-through
    alu = mux(op_c_movr, srcl_val, alu)

    # MOVI: immediate
    alu = mux(op_c_movi, imm, alu)

    # SETRET: ra = PC + off
    alu = mux(op_c_setret, pc_val + imm, alu)

    # SETC.EQ: (srcl == srcr) ? 1 : 0
    setc_eq_val = mux(srcl_val.eq(srcr_val), one64, z64)
    alu = mux(op_c_setc_eq, setc_eq_val, alu)

    # SETC.TGT: srcl
    alu = mux(op_c_setc_tgt, srcl_val, alu)

    # ADDTPC: PC + (imm<<12)
    pc_page = pc_val & 0xFFFF_FFFF_FFFF_F000
    alu = mux(op_addtpc, pc_page + imm, alu)

    # ADDI: srcl + imm
    alu = mux(op_addi, srcl_val + imm, alu)

    # SUBI: srcl - imm
    subi_val = srcl_val + ((~imm) + 1)
    alu = mux(op_subi, subi_val, alu)

    # ADDIW: 32-bit add with sign-extend
    addiw_val = (srcl_val.trunc(width=32) + imm.trunc(width=32)).sext(width=64)
    alu = mux(op_addiw, addiw_val, alu)

    # ADDW/ORW/ANDW/XORW: 32-bit ops with sign-extend
    addw_val = (srcl_val.trunc(width=32) + srcr_val.trunc(width=32)).sext(width=64)
    orw_val = (srcl_val.trunc(width=32) | srcr_val.trunc(width=32)).sext(width=64)
    andw_val = (srcl_val.trunc(width=32) & srcr_val.trunc(width=32)).sext(width=64)
    xorw_val = (srcl_val.trunc(width=32) ^ srcr_val.trunc(width=32)).sext(width=64)
    alu = mux(op_addw, addw_val, alu)
    alu = mux(op_orw, orw_val, alu)
    alu = mux(op_andw, andw_val, alu)
    alu = mux(op_xorw, xorw_val, alu)

    # CMP_EQ: (srcl == srcr) ? 1 : 0
    cmp_val = mux(srcl_val.eq(srcr_val), one64, z64)
    alu = mux(op_cmp_eq, cmp_val, alu)

    # HL.LUI: imm
    alu = mux(op_hl_lui, imm, alu)

    # CSEL: (srcp != 0) ? srcr : srcl
    csel_val = mux(srcp_val.ne(0), srcr_val, srcl_val)
    alu = mux(op_csel, csel_val, alu)

    # LWI / C.LWI: load word
    is_lwi = op_lwi | op_c_lwi
    lwi_addr = srcl_val + off
    is_load = mux(is_lwi, consts.one1, is_load)
    size = mux(is_lwi, m.ca_const(4, width=3), size)
    addr = mux(is_lwi, lwi_addr, addr)

    # SWI / C.SWI: store word
    store_addr_swi = srcr_val + off
    store_addr_c_swi = srcl_val + off
    store_addr = mux(op_swi, store_addr_swi, store_addr_c_swi)
    store_data = mux(op_swi, srcl_val, srcr_val)
    is_swi = op_swi | op_c_swi
    is_store = mux(is_swi, consts.one1, is_store)
    size = mux(is_swi, m.ca_const(4, width=3), size)
    addr = mux(is_swi, store_addr, addr)
    wdata = mux(is_swi, store_data, wdata)

    # SDI: store double word
    sdi_off = imm.shl(amount=3)
    sdi_addr = srcr_val + sdi_off
    is_store = mux(op_sdi, consts.one1, is_store)
    size = mux(op_sdi, m.ca_const(8, width=3), size)
    addr = mux(op_sdi, sdi_addr, addr)
    wdata = mux(op_sdi, srcl_val, wdata)

    # Pipeline regs: EX/MEM
    exmem.op.set(op, when=do_ex)
    exmem.len_bytes.set(len_bytes, when=do_ex)
    exmem.regdst.set(regdst, when=do_ex)
    exmem.alu.set(alu, when=do_ex)
    exmem.is_load.set(is_load, when=do_ex)
    exmem.is_store.set(is_store, when=do_ex)
    exmem.size.set(size, when=do_ex)
    exmem.addr.set(addr, when=do_ex)
    exmem.wdata.set(wdata, when=do_ex)
