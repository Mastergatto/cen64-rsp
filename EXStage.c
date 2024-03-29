/* ============================================================================
 *  EXStage.c: Execute stage.
 *
 *  RSPSIM: Reality Signal Processor SIMulator.
 *  Copyright (C) 2013, Tyler J. Stachecki.
 *  All rights reserved.
 *
 *  This file is subject to the terms and conditions defined in
 *  file 'LICENSE', which is part of this source code package.
 * ========================================================================= */
#include "Common.h"
#include "CPU.h"
#include "Definitions.h"
#include "EXStage.h"
#include "Memory.h"
#include "Pipeline.h"

#ifdef __cplusplus
#include <cassert>
#include <cstring>
#else
#include <assert.h>
#include <string.h>
#endif

/* ============================================================================
 *  Instruction: ADD{,U} (Add Signed/Unsigned)
 * ========================================================================= */
void
RSPADD(struct RSP *rsp, uint32_t rs, uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 11 & 0x1F;

  exdfLatch->result.data = rs + rt;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: ADDI{,U} (Add Immediate Signed/Unsigned)
 * ========================================================================= */
void
RSPADDI(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  int32_t imm = (int16_t) rdexLatch->iw;

  exdfLatch->result.data = rs + imm;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: AND (And)
 * ========================================================================= */
void
RSPAND(struct RSP *rsp, uint32_t rs, uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 11 & 0x1F;

  exdfLatch->result.data = rs & rt;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: ANDI (And Immediate)
 * ========================================================================= */
void
RSPANDI(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  uint32_t imm = rdexLatch->iw & 0xFFFF;

  exdfLatch->result.data = rs & imm;
  exdfLatch->result.dest = dest; 
}

/* ============================================================================
 *  Instruction: BEQ (Branch On Equal)
 * ========================================================================= */
void
RSPBEQ(struct RSP *rsp, uint32_t rs, uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPIFRDLatch *ifrdLatch = &rsp->pipeline.ifrdLatch;
  int32_t imm = (int16_t) rdexLatch->iw << 2;

  if (rs != rt)
    return;

  ifrdLatch->pc = rdexLatch->pc + imm;
  ifrdLatch->pc &= 0xFFC;
  ifrdLatch->pc |= 0x1000;
  rsp->didBranch = 1;
}

/* ============================================================================
 *  Instruction: BGEZ (Branch On Greater Than Or Equal To Zero)
 * ========================================================================= */
void
RSPBGEZ(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPIFRDLatch *ifrdLatch = &rsp->pipeline.ifrdLatch;
  int32_t imm = (int16_t) rdexLatch->iw << 2;

  if ((int32_t) rs < 0)
    return;

  ifrdLatch->pc = rdexLatch->pc + imm;
  ifrdLatch->pc &= 0xFFC;
  ifrdLatch->pc |= 0x1000;
  rsp->didBranch = 1;
}

/* ============================================================================
 *  Instruction: BGEZAL (Branch On Greater Than Or Equal To Zero And Link)
 * ========================================================================= */
void
RSPBGEZAL(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;
  struct RSPIFRDLatch *ifrdLatch = &rsp->pipeline.ifrdLatch;
  int32_t imm = (int16_t) rdexLatch->iw << 2;

  exdfLatch->result.data = ifrdLatch->pc;
  exdfLatch->result.dest = RSP_LINK_REGISTER;

  if ((int32_t) rs < 0)
    return;

  ifrdLatch->pc = rdexLatch->pc + imm;
  ifrdLatch->pc &= 0xFFC;
  ifrdLatch->pc |= 0x1000;
  rsp->didBranch = 1;
}

/* ============================================================================
 *  Instruction: BGTZ (Branch On Greater Than Zero)
 * ========================================================================= */
void
RSPBGTZ(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPIFRDLatch *ifrdLatch = &rsp->pipeline.ifrdLatch;
  int32_t imm = (int16_t) rdexLatch->iw << 2;

  if ((int32_t) rs <= 0)
    return;

  ifrdLatch->pc = rdexLatch->pc + imm;
  ifrdLatch->pc &= 0xFFC;
  ifrdLatch->pc |= 0x1000;
  rsp->didBranch = 1;
}

/* ============================================================================
 *  Instruction: BLEZ (Branch On Less Than Or Equal To Zero)
 * ========================================================================= */
void
RSPBLEZ(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPIFRDLatch *ifrdLatch = &rsp->pipeline.ifrdLatch;
  int32_t imm = (int16_t) rdexLatch->iw << 2;

  if ((int32_t) rs > 0)
    return;

  ifrdLatch->pc = rdexLatch->pc + imm;
  ifrdLatch->pc &= 0xFFC;
  ifrdLatch->pc |= 0x1000;
  rsp->didBranch = 1;
}

/* ============================================================================
 *  Instruction: BLTZ (Branch On Less Than Zero)
 * ========================================================================= */
void
RSPBLTZ(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPIFRDLatch *ifrdLatch = &rsp->pipeline.ifrdLatch;
  int32_t imm = (int16_t) rdexLatch->iw << 2;

  if ((int32_t) rs >= 0)
    return;

  ifrdLatch->pc = rdexLatch->pc + imm;
  ifrdLatch->pc &= 0xFFC;
  ifrdLatch->pc |= 0x1000;
  rsp->didBranch = 1;
}

/* ============================================================================
 *  Instruction: BLTZAL (Branch On Less Than Zero And Link)
 * ========================================================================= */
void
RSPBLTZAL(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;
  struct RSPIFRDLatch *ifrdLatch = &rsp->pipeline.ifrdLatch;
  int32_t imm = (int16_t) rdexLatch->iw << 2;

  exdfLatch->result.data = ifrdLatch->pc;
  exdfLatch->result.dest = RSP_LINK_REGISTER;

  if ((int32_t) rs >= 0)
    return;

  ifrdLatch->pc = rdexLatch->pc + imm;
  ifrdLatch->pc &= 0xFFC;
  ifrdLatch->pc |= 0x1000;
  rsp->didBranch = 1;
}

/* ============================================================================
 *  Instruction: BNE (Branch On Not Equal)
 * ========================================================================= */
void
RSPBNE(struct RSP *rsp, uint32_t rs, uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPIFRDLatch *ifrdLatch = &rsp->pipeline.ifrdLatch;
  int32_t imm = (int16_t) rdexLatch->iw << 2;

  if (rs == rt)
    return;

  ifrdLatch->pc = rdexLatch->pc + imm;
  ifrdLatch->pc &= 0xFFC;
  ifrdLatch->pc |= 0x1000;
  rsp->didBranch = 1;
}

/* ============================================================================
 *  Instruction: BREAK (Breakpoint)
 * ========================================================================= */
void
RSPBREAK(struct RSP *rsp, uint32_t unused(rs), uint32_t unused(rt)) {
  rsp->cp0.regs[SP_STATUS_REG] |= (SP_STATUS_HALT | SP_STATUS_BROKE);

  if (rsp->cp0.regs[SP_STATUS_REG] & SP_STATUS_INTR_BREAK)
    BusRaiseRCPInterrupt(rsp->bus, MI_INTR_SP);
}

/* ============================================================================
 *  Instruction: CFC2 (Move Control From Coprocessor 2 (VU))
 * ========================================================================= */
void
RSPCFC2(struct RSP *rsp, uint32_t unused(rs), uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned source = rdexLatch->iw >> 11 & 0x3;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned data;

  if (source == 3)
    source = 2;

  data = RSPGetFlags(
    &(rsp->cp2.vcolo) + (1 << source),
    &(rsp->cp2.vcohi) + (1 << source));

  exdfLatch->result.data = data;
  exdfLatch->result.dest = dest;
  
}
/* ============================================================================
 *  Instruction: CTC2 (Move Control To Coprocessor 2 (VU))
 * ========================================================================= */
void
RSPCTC2(struct RSP *rsp, uint32_t unused(rs), uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  unsigned dest = rdexLatch->iw >> 11 & 0x3;

  if (dest > 1)
    rt &= 0xFF;

  RSPSetFlags(
    &(rsp->cp2.vcolo) + (1 << dest),
    &(rsp->cp2.vcohi) + (1 << dest),
    rt);
}

/* ============================================================================
 *  Instruction: INV (Invalid Operation)
 * ========================================================================= */
void RSPINV(struct RSP *unused(rsp),
  uint32_t unused(rs), uint32_t unused(rt)) {}

/* ============================================================================
 *  Instruction: J (Jump)
 * ========================================================================= */
void
RSPJ(struct RSP *rsp, uint32_t unused(rs), uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPIFRDLatch *ifrdLatch = &rsp->pipeline.ifrdLatch;

  ifrdLatch->pc = (rdexLatch->iw & 0x3FF) << 2;
  ifrdLatch->pc |= 0x1000;
  rsp->didBranch = 1;
}

/* ============================================================================
 *  Instruction: JAL (Jump And Link)
 * ========================================================================= */
void
RSPJAL(struct RSP *rsp, uint32_t unused(rs), uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;
  struct RSPIFRDLatch *ifrdLatch = &rsp->pipeline.ifrdLatch;

  exdfLatch->result.data = ifrdLatch->pc;
  exdfLatch->result.dest = RSP_LINK_REGISTER;
  ifrdLatch->pc = (rdexLatch->iw & 0x3FF) << 2;
  ifrdLatch->pc |= 0x1000;
  rsp->didBranch = 1;
}

/* ============================================================================
 *  Instruction: JALR (Jump And Link Register)
 * ========================================================================= */
void
RSPJALR(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;
  struct RSPIFRDLatch *ifrdLatch = &rsp->pipeline.ifrdLatch;

  exdfLatch->result.data = ifrdLatch->pc;
  exdfLatch->result.dest = RSP_LINK_REGISTER;
  ifrdLatch->pc = rs;
  ifrdLatch->pc |= 0x1000;
  rsp->didBranch = 1;
}

/* ============================================================================
 *  Instruction: JR (Jump Register)
 * ========================================================================= */
void
RSPJR(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  struct RSPIFRDLatch *ifrdLatch = &rsp->pipeline.ifrdLatch;

  ifrdLatch->pc = rs & 0xFFC;
  ifrdLatch->pc |= 0x1000;
  rsp->didBranch = 1;
}

/* ============================================================================
 *  Instruction: LB (Load Byte)
 * ========================================================================= */
void
RSPLB(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;
  struct RSPDFWBLatch *dfwbLatch = &rsp->pipeline.dfwbLatch;

  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  int32_t offset = (int16_t) rdexLatch->iw;

  exdfLatch->result.dest = dest;
  exdfLatch->memoryData.target = &dfwbLatch->result.data;
  exdfLatch->memoryData.function = &LoadByte;
  exdfLatch->memoryData.offset = rs + offset;
}

/* ============================================================================
 *  Instruction: LBU (Load Byte Unsigned)
 * ========================================================================= */
void
RSPLBU(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;
  struct RSPDFWBLatch *dfwbLatch = &rsp->pipeline.dfwbLatch;

  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  int32_t offset = (int16_t) rdexLatch->iw;

  exdfLatch->result.dest = dest;
  exdfLatch->memoryData.target = &dfwbLatch->result.data;
  exdfLatch->memoryData.function = &LoadByteUnsigned;
  exdfLatch->memoryData.offset = rs + offset;
}

/* ============================================================================
 *  Instruction: LBV (Load Byte into Vector Register)
 * ========================================================================= */
void
RSPLBV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &LoadByteVector;
  exdfLatch->memoryData.offset = rs + offset;
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: LDV (Load Double into Vector Register)
 * ========================================================================= */
void
RSPLDV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &LoadDoubleVector;
  exdfLatch->memoryData.offset = rs + (offset << 3);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: LFV (Load Packed Fourth into Vector Register)
 * ========================================================================= */
void
RSPLFV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &LoadPackedFourthVector;
  exdfLatch->memoryData.offset = rs + (offset << 4);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: LH (Load Halfword)
 * ========================================================================= */
void
RSPLH(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;
  struct RSPDFWBLatch *dfwbLatch = &rsp->pipeline.dfwbLatch;

  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  int32_t offset = (int16_t) rdexLatch->iw;

  exdfLatch->result.dest = dest;
  exdfLatch->memoryData.target = &dfwbLatch->result.data;
  exdfLatch->memoryData.function = &LoadHalf;
  exdfLatch->memoryData.offset = rs + offset;
}

/* ============================================================================
 *  Instruction: LHU (Load Halfword Unsigned)
 * ========================================================================= */
void
RSPLHU(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;
  struct RSPDFWBLatch *dfwbLatch = &rsp->pipeline.dfwbLatch;

  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  int32_t offset = (int16_t) rdexLatch->iw;

  exdfLatch->result.dest = dest;
  exdfLatch->memoryData.target = &dfwbLatch->result.data;
  exdfLatch->memoryData.function = &LoadHalfUnsigned;
  exdfLatch->memoryData.offset = rs + offset;
}

/* ============================================================================
 *  Instruction: LHV (Load Packed Half into Vector Register)
 * ========================================================================= */
void
RSPLHV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &LoadPackedHalfVector;
  exdfLatch->memoryData.offset = rs + (offset << 4);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: LLV (Load Long into Vector Register)
 * ========================================================================= */
void
RSPLLV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &LoadLongVector;
  exdfLatch->memoryData.offset = rs + (offset << 2);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: LPV (Load Packed Bytes into Vector Register)
 * ========================================================================= */
void
RSPLPV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &LoadPackedByteVector;
  exdfLatch->memoryData.offset = rs + (offset << 3);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: LQV (Load Quad into Vector Register)
 * ========================================================================= */
void
RSPLQV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &LoadQuadVector;
  exdfLatch->memoryData.offset = rs + (offset << 4);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: LRV (Load Quad (Rest) into Vector Register)
 * ========================================================================= */
void
RSPLRV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &LoadRestVector;
  exdfLatch->memoryData.offset = rs + (offset << 4);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: LSV (Load Short into Vector Register)
 * ========================================================================= */
void
RSPLSV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &LoadShortVector;
  exdfLatch->memoryData.offset = rs + (offset << 1);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: LTV (Load Transpose into Vector Register)
 * ========================================================================= */
void
RSPLTV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  assert((dest & 7) == 0 && "STV: Invalid `vt` for transpose specified.");

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &LoadTransposeVector;
  exdfLatch->memoryData.offset = rs + (offset << 4);
  exdfLatch->memoryData.element = element;
  exdfLatch->memoryData.cp2 = &rsp->cp2;
}

/* ============================================================================
 *  Instruction: LUI (Load Upper Immediate)
 * ========================================================================= */
void
RSPLUI(struct RSP *rsp, uint32_t unused(rs), uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  uint32_t imm = rdexLatch->iw & 0xFFFF;

  exdfLatch->result.data = imm << 16;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: LUV (Load Unsigned Packed into Vector Register)
 * ========================================================================= */
void
RSPLUV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &LoadPackedVector;
  exdfLatch->memoryData.offset = rs + (offset << 3);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: LW (Load Word)
 * ========================================================================= */
void
RSPLW(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;
  struct RSPDFWBLatch *dfwbLatch = &rsp->pipeline.dfwbLatch;

  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  int32_t offset = (int16_t) rdexLatch->iw;

  exdfLatch->result.dest = dest;
  exdfLatch->memoryData.target = &dfwbLatch->result.data;
  exdfLatch->memoryData.function = &LoadWord;
  exdfLatch->memoryData.offset = rs + offset;
}

/* ============================================================================
 *  Instruction: MFC2 (Move From Coprocessor 2 (VU))
 * ========================================================================= */
void
RSPMFC2(struct RSP *rsp, uint32_t unused(rs), uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned source = rdexLatch->iw >> 11 & 0x1F;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  uint8_t slices[16], byte1, byte2;

  /* Not treated the same? */
  assert(element != 0xF);

  int16_t data;
  CopyVectorSlices(rsp->cp2.regs[source].slices, slices);
  byte2 = slices[(element + 1) & 0xF];
  byte1 = slices[element];

  data = byte2 | (byte1 << 8);
  exdfLatch->result.data = (int32_t) data;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: MTC2 (Move To Coprocessor 2 (VU))
 * ========================================================================= */
void
RSPMTC2(struct RSP *rsp, uint32_t unused(rs), uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 11 & 0x1F;

  assert((element & 0x1) != 1);

  rsp->cp2.regs[dest].slices[element >> 1] = rt;
}

/* ============================================================================
 *  Instruction: NOP (No Operation)
 * ========================================================================= */
void
RSPNOP(struct RSP *unused(rsp), uint32_t unused(rs), uint32_t unused(rt)) {
}

/* ============================================================================
 *  Instruction: NOR (Nor)
 * ========================================================================= */
void
RSPNOR(struct RSP *rsp, uint32_t rs, uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 11 & 0x1F;

  exdfLatch->result.data = ~(rs | rt);
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: OR (OR)
 * ========================================================================= */
void
RSPOR(struct RSP *rsp, uint32_t rs, uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 11 & 0x1F;

  exdfLatch->result.data = rs | rt;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: ORI (OR Immediate)
 * ========================================================================= */
void
RSPORI(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  uint32_t imm = rdexLatch->iw & 0xFFFF;

  exdfLatch->result.data = rs | imm;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: SB (Store Byte)
 * ========================================================================= */
void
RSPSB(struct RSP *rsp, uint32_t rs, uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  int32_t offset = (int16_t) rdexLatch->iw;

  exdfLatch->memoryData.data = rt;
  exdfLatch->memoryData.function = &StoreByte;
  exdfLatch->memoryData.offset = rs + offset;
}

/* ============================================================================
 *  Instruction: SBV (Store Byte from Vector Register)
 * ========================================================================= */
void
RSPSBV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &StoreByteVector;
  exdfLatch->memoryData.offset = rs + offset;
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: SDV (Store Double from Vector Register)
 * ========================================================================= */
void
RSPSDV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &StoreDoubleVector;
  exdfLatch->memoryData.offset = rs + (offset << 3);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: SFV (Store Packed Fourth from Vector Register)
 * ========================================================================= */
void
RSPSFV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &StorePackedFourthVector;
  exdfLatch->memoryData.offset = rs + (offset << 4);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: SH (Store Half)
 * ========================================================================= */
void
RSPSH(struct RSP *rsp, uint32_t rs, uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  int32_t offset = (int16_t) rdexLatch->iw;

  exdfLatch->memoryData.data = rt;
  exdfLatch->memoryData.function = &StoreHalf;
  exdfLatch->memoryData.offset = rs + offset;
}

/* ============================================================================
 *  Instruction: SHV (Store Packed Half from Vector Register)
 * ========================================================================= */
void
RSPSHV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &StorePackedHalfVector;
  exdfLatch->memoryData.offset = rs + (offset << 4);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: SLL (Shift Left Logical)
 * ========================================================================= */
void
RSPSLL(struct RSP *rsp, uint32_t unused(rs), uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 11 & 0x1F;
  uint32_t sa = rdexLatch->iw >> 6 & 0x1F;

  exdfLatch->result.data = rt << sa;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: SLLV  (Shift Left Logical Variable)
 * ========================================================================= */
void
RSPSLLV(struct RSP *rsp, uint32_t rs, uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 11 & 0x1F;
  uint32_t sa = rs & 0x1F;

  exdfLatch->result.data = rt << sa;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: SLV (Store Long from Vector Register)
 * ========================================================================= */
void
RSPSLV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &StoreLongVector;
  exdfLatch->memoryData.offset = rs + (offset << 2);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: SLT (Set On Less Than)
 * ========================================================================= */
void
RSPSLT(struct RSP *rsp, uint32_t rs, uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 11 & 0x1F;

  exdfLatch->result.data = ((int32_t) rs < (int32_t) rt) ? 1 : 0;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: SLTI (Set On Less Than Immediate)
 * ========================================================================= */
void
RSPSLTI(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  int32_t imm = (int16_t) rdexLatch->iw;

  exdfLatch->result.data = ((int32_t) rs < imm) ? 1 : 0;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: SLTIU (Set On Less Than Immediate Unsigned)
 * ========================================================================= */
void
RSPSLTIU(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  uint32_t imm = (int32_t) ((int16_t) rdexLatch->iw);
  
  exdfLatch->result.data = (rs < imm) ? 1 : 0;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: SLTU (Set On Less Than Unsigned)
 * ========================================================================= */
void
RSPSLTU(struct RSP *rsp, uint32_t rs, uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 11 & 0x1F;

  exdfLatch->result.data = (rs < rt) ? 1 : 0;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: SPV (Store Packed Bytes from Vector Register)
 * ========================================================================= */
void
RSPSPV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &StorePackedByteVector;
  exdfLatch->memoryData.offset = rs + (offset << 3);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: SQV (Store Quad from Vector Register)
 * ========================================================================= */
void
RSPSQV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &StoreQuadVector;
  exdfLatch->memoryData.offset = rs + (offset << 4);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: SRA (Shift Right Arithmetic)
 * ========================================================================= */
void
RSPSRA(struct RSP *rsp, uint32_t unused(rs), uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 11 & 0x1F;
  uint32_t sa = rdexLatch->iw >> 6 & 0x1F;

  exdfLatch->result.data = (int32_t) rt >> sa;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: SRAV (Shift Right Arithmetic Variable)
 * ========================================================================= */
void
RSPSRAV(struct RSP *rsp, uint32_t rs, uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 11 & 0x1F;
  unsigned sa = rs & 0x1F;

  exdfLatch->result.data = (int32_t) rt >> sa;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: SRL (Shift Right Logical)
 * ========================================================================= */
void
RSPSRL(struct RSP *rsp, uint32_t unused(rs), uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 11 & 0x1F;
  uint32_t sa = rdexLatch->iw >> 6 & 0x1F;

  exdfLatch->result.data = rt >> sa;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: SRLV (Shift Right Logical Variable)
 * ========================================================================= */
void
RSPSRLV(struct RSP *rsp, uint32_t rs, uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 11 & 0x1F;
  uint32_t sa = rs & 0x1F;

  exdfLatch->result.data = rt >> sa;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: SRV (Store Quad (Rest) from Vector Register)
 * ========================================================================= */
void
RSPSRV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &StoreRestVector;
  exdfLatch->memoryData.offset = rs + (offset << 4);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: SSV (Store Short from Vector Register)
 * ========================================================================= */
void
RSPSSV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &StoreShortVector;
  exdfLatch->memoryData.offset = rs + (offset << 1);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: STV (Store Transpose into Vector Register)
 * ========================================================================= */
void
RSPSTV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;
  unsigned i, start;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  assert((dest & 7) == 0 && "STV: Invalid `vt` for transpose specified.");
  assert((element & 0x1) == 0 && "Element references odd byte of slice?");

  exdfLatch->memoryData.target = &rsp->cp2.regs[NUM_RSP_VP_REGISTERS];
  exdfLatch->memoryData.function = &StoreTransposeVector;
  exdfLatch->memoryData.offset = rs + (offset << 4);
  exdfLatch->memoryData.element = element;

  /* Collect all the pieces. */
  start = element >> 1;
  for (i = 0; i < 8; start = (start + 1) & 7, i++) {
    uint16_t slice = rsp->cp2.regs[dest + start].slices[i];
    rsp->cp2.transposeVector.slices[i] = slice;
  }
}

/* ============================================================================
 *  Instruction: SUV (Store Unsigned Packed from Vector Register)
 * ========================================================================= */
void
RSPSUV(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned element = rdexLatch->iw >> 7 & 0xF;
  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  unsigned offset = rdexLatch->iw & 0x7F;
  offset |= -(offset & 0x0040);

  exdfLatch->memoryData.target = &rsp->cp2.regs[dest];
  exdfLatch->memoryData.function = &StorePackedVector;
  exdfLatch->memoryData.offset = rs + (offset << 3);
  exdfLatch->memoryData.element = element;
}

/* ============================================================================
 *  Instruction: SUB{,U} (Subtract Signed/Unsigned)
 * ========================================================================= */
void
RSPSUB(struct RSP *rsp, uint32_t rs, uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 11 & 0x1F;

  exdfLatch->result.data = rs - rt;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: SW (Store Word)
 * ========================================================================= */
void
RSPSW(struct RSP *rsp, uint32_t rs, uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  int32_t offset = (int16_t) rdexLatch->iw;

  exdfLatch->memoryData.data = rt;
  exdfLatch->memoryData.function = &StoreWord;
  exdfLatch->memoryData.offset = rs + offset;
}

/* ============================================================================
 *  Instruction: SWV (Store Wrapped from Vector Register)
 * ========================================================================= */
void
RSPSWV(struct RSP *unused(rsp), uint32_t unused(rs), uint32_t unused(rt)) {
  debug("Unimplemented function: SWV.");
}

/* ============================================================================
 *  Instruction: XOR (XOR)
 * ========================================================================= */
void
RSPXOR(struct RSP *rsp, uint32_t rs, uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 11 & 0x1F;
  exdfLatch->result.data = rs ^ rt;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: XORI (XOR Immediate)
 * ========================================================================= */
void
RSPXORI(struct RSP *rsp, uint32_t rs, uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned dest = rdexLatch->iw >> 16 & 0x1F;
  uint32_t imm = rdexLatch->iw & 0xFFFF;

  exdfLatch->result.data = rs ^ imm;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  RSPEXStage: Invokes the appropriate functional unit.
 * ========================================================================= */
void
RSPEXStage(struct RSP *rsp,
  unsigned rsForwardingRegister, unsigned rtForwardingRegister) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  /* Always invalidate results. */
  exdfLatch->result.dest = 0;

  /* Don't execute anything if decode killed us. */
  if (rdexLatch->opcode.id != RSP_OPCODE_INV) {
    const struct RSPDFWBLatch *dfwbLatch = &rsp->pipeline.dfwbLatch;
    uint32_t rs, rt, temp = rsp->regs[dfwbLatch->result.dest];

    rsp->didBranch = 0;

    /* Forward results from DC/WB into the register file (RF). */
    /* Copy/restore value to prevent the need for branches. */
    rsp->regs[dfwbLatch->result.dest] = dfwbLatch->result.data;
    rsp->regs[RSP_REGISTER_ZERO] = 0;

    rs = rsp->regs[rsForwardingRegister];
    rt = rsp->regs[rtForwardingRegister];
    rsp->regs[dfwbLatch->result.dest] = temp;

    RSPScalarFunctionTable[rdexLatch->opcode.id](rsp, rs, rt);
  }

#ifndef NDEBUG
  rsp->pipeline.counts[rdexLatch->opcode.id]++;
#endif
}

