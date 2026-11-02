"""Bus-exact instruction timing for generated code (SNESRECOMP_EXACT_BUS=1 at generation).

The interpreter tier charges an instruction the region speed of each bus transfer
it makes (opcode and operand fetches, pointer, data and stack bytes) plus 6
master clocks per remaining (internal) cycle, and lets an I/O register access
see the clock of its own bus cycle (interp_bridge.c, bridge_timing_bus /
bridge_note_access). The default emitter folds a whole block into
`cycles x code speed`, so every WRAM or I/O access is billed at code speed, and
the block's clocks are all charged before any of its accesses.

With this switch on, every instruction is its own CFG block (the deadline is
checked between instructions, as the interpreter does) and each instruction
charges exactly what the interpreter charges for it:

    master = 6 x cycles + sum over transfers of (region speed - 6)

which equals `sum of transfer speeds + 6 x internal cycles` while the
transfers do not outnumber the cycles. Instructions that touch memory through
an operand charge after their effect, with `g_aot_access_prefix` holding the
clocks from the instruction's start to its data access, so an I/O register
sees the same clock it sees under the interpreter. Everything else charges
before its effect (a call's callee and a branch target come after it).
"""

import os
from typing import List, Optional, Tuple

import snes65816 as _d
from snes65816 import (
    IMP, ACC, IMM, DP, DP_X, DP_Y, ABS, ABS_X, ABS_Y, LONG, LONG_X, REL,
    REL16, STK, INDIR, INDIR_X, INDIR_Y, INDIR_LY, INDIR_L, INDIR_DPX,
    DP_INDIR, STK_IY,
)
from snes_cycles import instr_static_cycles, dp_add

from v2.ir import SegKind


def enabled() -> bool:
    return os.environ.get('SNESRECOMP_EXACT_BUS') == '1'


_STORE = {'STA', 'STX', 'STY', 'STZ'}
_RMW = {'ASL', 'LSR', 'ROL', 'ROR', 'INC', 'DEC', 'TSB', 'TRB'}
_XW = {'LDX', 'LDY', 'STX', 'STY', 'CPX', 'CPY'}
_MEM = {'LDA', 'LDX', 'LDY', 'ADC', 'SBC', 'AND', 'ORA', 'EOR', 'CMP',
        'CPX', 'CPY', 'BIT'} | _STORE | _RMW
_NO_OPERAND_MODES = {IMP, ACC, IMM, REL, REL16}
# Pointer bytes the indirect modes read before the data (bank $00).
_PTR_BYTES = {DP_INDIR: 2, INDIR_DPX: 2, INDIR_Y: 2, INDIR_L: 3,
              INDIR_LY: 3, STK_IY: 2}
_READ_XCROSS_MODES = {ABS_X, ABS_Y, INDIR_Y}


def _stack_bytes(op: int, mnem: str, m: int, x: int) -> int:
    if mnem in ('PHA', 'PLA'):
        return 1 if m else 2
    if mnem in ('PHX', 'PHY', 'PLX', 'PLY'):
        return 1 if x else 2
    if mnem in ('PHB', 'PLB', 'PHK', 'PHP', 'PLP'):
        return 1
    if mnem in ('PHD', 'PLD', 'PEA', 'PEI', 'PER', 'JSR', 'RTS'):
        return 2
    if mnem in ('JSL', 'RTL'):
        return 3
    if mnem == 'RTI':
        return 4                       # native mode: P, PC, PB
    if mnem in ('BRK', 'COP'):
        return 4
    return 0


def _rs(bank_expr: str, addr_expr: str) -> str:
    return (f"cpu_region_speed(((uint32)({bank_expr}) << 16) | "
            f"(uint32)(uint16)({addr_expr}))")


def _rs0(addr_expr: str) -> str:
    return f"cpu_region_speed((uint32)(uint16)({addr_expr}))"


class Plan:
    """What one instruction charges and where."""

    def __init__(self):
        self.cycles_const = 0
        self.cycles_dyn: List[str] = []    # C expressions, each 0 or 1
        self.master_const = 0              # clocks above 6 x cycles
        self.master_dyn: List[str] = []    # C expressions in clocks
        self.after = False                 # charge after the effect
        self.prefix_expr: Optional[str] = None   # clocks to the data access
        self.wprefix_expr: Optional[str] = None  # to the write of an RMW

    def lines(self, tag: str) -> Tuple[List[str], List[str]]:
        """(lines before the effect, lines after it)."""
        cyc = " + ".join([str(self.cycles_const)] + self.cycles_dyn)
        extra = " + ".join([str(self.master_const)] + self.master_dyn)
        decl = [f"uint32 _bc{tag} = {cyc};",
                f"uint32 _bx{tag} = 6u * _bc{tag} + {extra};"]
        charge = [f"cpu->cycles += _bc{tag};",
                  f"cpu->master_cycles += _bx{tag};"]
        if not self.after:
            return decl + charge, []
        pre = decl + [f"g_aot_access_prefix = {self.prefix_expr};",
                      f"g_aot_access_wprefix = {self.wprefix_expr};"]
        post = charge + ["g_aot_access_prefix = AOT_NO_ACCESS;"]
        return pre, post


def _first_seg(ir_ops):
    for op in ir_ops:
        seg = getattr(op, 'seg', None)
        if seg is not None:
            return seg
    return None


def plan(insn, ir_ops, code_speed: Tuple[str, Optional[int]]) -> Plan:
    from v2.codegen import _segref_addr_expr

    p = Plan()
    op, mnem, mode = insn.opcode, insn.mnem, insn.mode
    m = 1 if (getattr(insn, 'm_flag', 1) & 1) else 0
    x = 1 if (getattr(insn, 'x_flag', 1) & 1) else 0
    cs_expr, cs_const = code_speed
    cs = str(cs_const) if cs_const is not None else f"({cs_expr})"

    p.cycles_const = instr_static_cycles(op, m, x)
    if dp_add(op):
        p.cycles_dyn.append("((cpu->D & 0xFF) ? 1u : 0u)")

    # Opcode and operand fetches, at the code region's speed.
    if cs_const is not None:
        p.master_const += insn.length * (cs_const - 6)
    else:
        p.master_dyn.append(f"{insn.length}u * ({cs} - 6u)")

    sb = _stack_bytes(op, mnem, m, x)
    if sb:
        p.master_dyn.append(f"{sb}u * ({_rs0('cpu->S')} - 6u)")
    if mnem in ('BRK', 'COP'):
        p.master_dyn.append(f"2u * ({_rs0('0xFFE6')} - 6u)")

    operand = insn.operand if isinstance(insn.operand, int) else 0
    if op == 0x6C:                     # JMP (abs): pointer in bank $00
        p.master_dyn.append(f"2u * ({_rs0(hex(operand & 0xFFFF))} - 6u)")
        return p
    if op == 0xDC:                     # JML [abs]
        p.master_dyn.append(f"3u * ({_rs0(hex(operand & 0xFFFF))} - 6u)")
        return p
    if op in (0x7C, 0xFC):             # JMP/JSR (abs,X): pointer in PB
        p.master_dyn.append(
            f"2u * ({_rs('cpu->PB', f'{operand & 0xFFFF:#06x} + cpu->X')} - 6u)")
        return p
    if mnem == 'PEI':
        p.master_dyn.append(
            f"2u * ({_rs0(f'cpu->D + {operand & 0xFF:#04x}')} - 6u)")
        return p
    if mnem in ('MVN', 'MVP'):
        bm = next(o for o in ir_ops if hasattr(o, 'src_bank'))
        src, dst = bm.src_bank, bm.dst_bank
        p.master_dyn.append(f"({_rs(hex(src), 'cpu->X')} - 6u)")
        p.master_dyn.append(f"({_rs(hex(dst), 'cpu->Y')} - 6u)")
        return p
    if mnem not in _MEM or mode in _NO_OPERAND_MODES:
        return p

    seg = _first_seg(ir_ops)
    if seg is None:
        return p
    width = (1 if x else 2) if mnem in _XW else (1 if m else 2)
    nbytes = width * (2 if mnem in _RMW else 1)
    prefix = [f"{insn.length}u * {cs}"]

    nptr = _PTR_BYTES.get(mode, 0)
    if nptr:
        off = seg.offset & 0xFF
        if mode == STK_IY:
            pa = f"cpu->S + {off:#04x}"
        elif mode == INDIR_DPX:
            pa = f"cpu->D + {off:#04x} + cpu->X"
        else:
            pa = f"cpu->D + {off:#04x}"
        p.master_dyn.append(f"{nptr}u * ({_rs0(pa)} - 6u)")
        prefix.append(f"{nptr}u * {_rs0(pa)}")

    bank, addr = _segref_addr_expr(seg)
    ds = _rs(bank, addr)
    p.master_dyn.append(f"{nbytes}u * ({ds} - 6u)")

    if x and mnem not in _STORE and mnem not in _RMW and mode in _READ_XCROSS_MODES:
        if mode == INDIR_Y:
            base = f"cpu_read16(cpu, 0x00, (uint16)(cpu->D + {seg.offset & 0xFF:#04x}))"
            idx = "cpu->Y"
        else:
            base = f"{operand & 0xFFFF:#06x}u"
            idx = "cpu->X" if mode == ABS_X else "cpu->Y"
        p.cycles_dyn.append(
            f"((((uint32)({base}) >> 8) != (((uint32)({base}) + (uint32){idx}) >> 8)) ? 1u : 0u)")

    p.after = True
    p.prefix_expr = " + ".join(prefix)
    if mnem in _RMW:
        p.wprefix_expr = f"{p.prefix_expr} + {width}u * {ds}"
    else:
        p.wprefix_expr = p.prefix_expr
    return p
