from __future__ import annotations

from dataclasses import dataclass

from pycircuit import Circuit, Wire, jit_inline

from .isa import (
    OP_ADDTPC,
    OP_ADDI,
    OP_ADDIW,
    OP_ADDW,
    OP_ANDW,
    OP_BSTART_STD_CALL,
    OP_CMP_EQ,
    OP_CSEL,
    OP_C_BSTOP,
    OP_C_BSTART_COND,
    OP_C_BSTART_STD,
    OP_C_LWI,
    OP_C_MOVI,
    OP_C_MOVR,
    OP_C_SETC_EQ,
    OP_C_SETC_TGT,
    OP_C_SETRET,
    OP_C_SWI,
    OP_EBREAK,
    OP_FENTRY,
    OP_HL_LUI,
    OP_INVALID,
    OP_LWI,
    OP_ORW,
    OP_SDI,
    OP_SUBI,
    OP_SWI,
    OP_XORW,
    REG_INVALID,
)
from .util import masked_eq


@dataclass(frozen=True)
class Decode:
    op: Wire
    len_bytes: Wire
    regdst: Wire
    srcl: Wire
    srcr: Wire
    srcp: Wire
    imm: Wire


@jit_inline
def decode_window(m: Circuit, window: Wire) -> Decode:
    c = m.const_wire

    zero3 = c(0, width=3)
    zero64 = c(0, width=64)
    reg_invalid = c(REG_INVALID, width=6)

    insn16 = window.trunc(width=16)
    insn32 = window.trunc(width=32)
    insn48 = window.trunc(width=48)

    low4 = insn16[0:4]
    is_hl = low4 == 0xE

    is32 = insn16[0]
    in32 = (~is_hl) & is32
    in16 = (~is_hl) & (~is32)

    rd32 = insn32[7:12]
    rs1_32 = insn32[15:20]
    rs2_32 = insn32[20:25]
    srcp_32 = insn32[27:32]

    imm12_u64 = insn32[20:32]
    imm12_s64 = insn32[20:32].sext(width=64)

    imm20_s64 = insn32[12:32].sext(width=64)

    # SWI simm12 is split: {insn32[11:7], insn32[31:25]}.
    swi_lo5 = insn32[7:12]
    swi_hi7 = insn32[25:32]
    simm12_raw = swi_lo5.zext(width=12).shl(amount=7) | swi_hi7.zext(width=12)
    simm12_s64 = simm12_raw.sext(width=64)
    simm17_s64 = insn32[15:32].sext(width=64)

    # HL.LUI immediate packing (48-bit):
    # pfx = insn48[15:0]; main = insn48[47:16]
    pfx16 = insn48.trunc(width=16)
    main32 = insn48[16:48]
    imm_hi12 = pfx16[4:16]
    imm_lo20 = main32[12:32]
    imm32 = imm_hi12.zext(width=32).shl(amount=20) | imm_lo20.zext(width=32)
    imm_hl_lui = imm32.sext(width=64)

    rd_hl = main32[7:12]

    # 16-bit fields.
    rd16 = insn16[11:16]
    rs16 = insn16[6:11]
    # Immediate fields:
    # - simm5_11_s5: bits[15:11] (used by loads/stores)
    # - simm5_6_s5: bits[10:6] (used by C.MOVI / C.SETRET)
    simm5_11_s64 = insn16[11:16].sext(width=64)
    simm5_6_s64 = insn16[6:11].sext(width=64)
    simm12_s64_c = insn16[4:16].sext(width=64)
    uimm5 = insn16[6:11]
    brtype = insn16[11:14]

    op = c(OP_INVALID, width=6)
    len_bytes = zero3
    regdst = reg_invalid
    srcl = reg_invalid
    srcr = reg_invalid
    srcp = reg_invalid
    imm = zero64

    # --- 16-bit decode (reverse priority; C.BSTOP highest) ---
    cond = in16 & masked_eq(insn16, mask=0x003F, match=0x0016)
    if cond:
        op = OP_C_MOVI
        len_bytes = 2
        regdst = rd16
        imm = simm5_6_s64

    cond = in16 & masked_eq(insn16, mask=0xF83F, match=0x5016)
    if cond:
        op = OP_C_SETRET
        len_bytes = 2
        regdst = 10  # ra
        imm = uimm5.zext(width=6).shl(amount=1)

    cond = in16 & masked_eq(insn16, mask=0x003F, match=0x002A)
    if cond:
        op = OP_C_SWI
        len_bytes = 2
        srcl = rs16
        srcr = 24
        imm = simm5_11_s64

    cond = in16 & masked_eq(insn16, mask=0x003F, match=0x000A)
    if cond:
        op = OP_C_LWI
        len_bytes = 2
        srcl = rs16
        imm = simm5_11_s64

    cond = in16 & masked_eq(insn16, mask=0x003F, match=0x0006)
    if cond:
        op = OP_C_MOVR
        len_bytes = 2
        regdst = rd16
        srcl = rs16

    cond = in16 & masked_eq(insn16, mask=0x003F, match=0x0026)
    if cond:
        op = OP_C_SETC_EQ
        len_bytes = 2
        srcl = rs16
        srcr = rd16

    cond = in16 & masked_eq(insn16, mask=0xF83F, match=0x001C)
    if cond:
        op = OP_C_SETC_TGT
        len_bytes = 2
        srcl = rs16

    cond = in16 & masked_eq(insn16, mask=0x000F, match=0x0004)
    if cond:
        op = OP_C_BSTART_COND
        len_bytes = 2
        imm = simm12_s64_c.shl(amount=1)

    cond = in16 & masked_eq(insn16, mask=0xC7FF, match=0x0000)
    if cond:
        op = OP_C_BSTART_STD
        len_bytes = 2
        imm = brtype

    cond = in16 & masked_eq(insn16, mask=0xFFFF, match=0x0000)
    if cond:
        op = OP_C_BSTOP
        len_bytes = 2

    # --- 32-bit decode (reverse priority; EBREAK highest) ---
    cond = in32 & masked_eq(insn32, mask=0x0000707F, match=0x00004025)
    if cond:
        op = OP_XORW
        len_bytes = 4
        regdst = rd32
        srcl = rs1_32
        srcr = rs2_32

    cond = in32 & masked_eq(insn32, mask=0x0000707F, match=0x00002025)
    if cond:
        op = OP_ANDW
        len_bytes = 4
        regdst = rd32
        srcl = rs1_32
        srcr = rs2_32

    cond = in32 & masked_eq(insn32, mask=0x0000707F, match=0x00003025)
    if cond:
        op = OP_ORW
        len_bytes = 4
        regdst = rd32
        srcl = rs1_32
        srcr = rs2_32

    cond = in32 & masked_eq(insn32, mask=0x0000707F, match=0x00000025)
    if cond:
        op = OP_ADDW
        len_bytes = 4
        regdst = rd32
        srcl = rs1_32
        srcr = rs2_32

    cond = in32 & masked_eq(insn32, mask=0x0000707F, match=0x00000077)
    if cond:
        op = OP_CSEL
        len_bytes = 4
        regdst = rd32
        srcl = rs1_32
        srcr = rs2_32
        srcp = srcp_32

    cond = in32 & masked_eq(insn32, mask=0x0000707F, match=0x00002059)
    if cond:
        op = OP_SWI
        len_bytes = 4
        srcl = rs1_32
        srcr = rs2_32
        imm = simm12_s64

    cond = in32 & masked_eq(insn32, mask=0x0000707F, match=0x00003059)
    if cond:
        op = OP_SDI
        len_bytes = 4
        srcl = rs1_32
        srcr = rs2_32
        imm = simm12_s64

    cond = in32 & masked_eq(insn32, mask=0x0000707F, match=0x00002019)
    if cond:
        op = OP_LWI
        len_bytes = 4
        regdst = rd32
        srcl = rs1_32
        imm = imm12_s64

    cond = in32 & masked_eq(insn32, mask=0x0000707F, match=0x00000035)
    if cond:
        op = OP_ADDIW
        len_bytes = 4
        regdst = rd32
        srcl = rs1_32
        imm = imm12_u64

    cond = in32 & masked_eq(insn32, mask=0x0000707F, match=0x00000015)
    if cond:
        op = OP_ADDI
        len_bytes = 4
        regdst = rd32
        srcl = rs1_32
        imm = imm12_u64

    cond = in32 & masked_eq(insn32, mask=0x0000707F, match=0x00001015)
    if cond:
        op = OP_SUBI
        len_bytes = 4
        regdst = rd32
        srcl = rs1_32
        imm = imm12_u64

    cond = in32 & masked_eq(insn32, mask=0xF800707F, match=0x00000045)
    if cond:
        op = OP_CMP_EQ
        len_bytes = 4
        regdst = rd32
        srcl = rs1_32
        srcr = rs2_32

    cond = in32 & masked_eq(insn32, mask=0x0000707F, match=0x00000041)
    if cond:
        op = OP_FENTRY
        len_bytes = 4

    cond = in32 & masked_eq(insn32, mask=0xF0FFFFFF, match=0x0010102B)
    if cond:
        op = OP_EBREAK
        len_bytes = 4

    cond = in32 & masked_eq(insn32, mask=0x0000007F, match=0x00000007)
    if cond:
        op = OP_ADDTPC
        len_bytes = 4
        regdst = rd32
        imm = imm20_s64.shl(amount=12)

    cond = in32 & masked_eq(insn32, mask=0x00007FFF, match=0x00004001)
    if cond:
        op = OP_BSTART_STD_CALL
        len_bytes = 4
        imm = simm17_s64.shl(amount=1)

    # --- 48-bit HL.LUI (highest priority overall) ---
    cond = is_hl & masked_eq(insn48, mask=0x0000007F000F, match=0x00000017000E)
    if cond:
        op = OP_HL_LUI
        len_bytes = 6
        regdst = rd_hl
        imm = imm_hl_lui

    return Decode(op=op, len_bytes=len_bytes, regdst=regdst, srcl=srcl, srcr=srcr, srcp=srcp, imm=imm)
