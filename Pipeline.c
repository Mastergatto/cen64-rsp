/* ============================================================================
 *  Pipeline.c: Processor pipeline.
 *
 *  RSPSIM: Reality Signal Processor SIMulator.
 *  Copyright (C) 2013, Tyler J. Stachecki.
 *  All rights reserved.
 *
 *  This file is subject to the terms and conditions defined in
 *  file 'LICENSE', which is part of this source code package.
 * ========================================================================= */
#include "Common.h"
#include "CP0.h"
#include "CP2.h"
#include "CPU.h"
#include "Decoder.h"
#include "DFStage.h"
#include "EXStage.h"
#include "IFStage.h"
#include "Opcodes.h"
#include "Pipeline.h"
#include "RDStage.h"
#include "WBStage.h"

#ifdef __cplusplus
#include <cassert>
#include <cstring>
#else
#include <assert.h>
#include <string.h>
#endif

/* ============================================================================
 *  IsLoadStoreStall: Determines if a scalar load/store condition is present.
 * ========================================================================= */
static bool
IsLoadStoreStall(uint32_t rdInfoFlags, uint32_t dfInfoFlags) {
  uint32_t copsOpcodeInfoMask = OPCODE_INFO_CP0 | OPCODE_INFO_CP2;
  uint32_t dfStallConditionMask = OPCODE_INFO_LOAD | copsOpcodeInfoMask;
  uint32_t rdStallConditionMask = OPCODE_INFO_STORE | copsOpcodeInfoMask;

  return (dfInfoFlags & dfStallConditionMask) &&
         (rdInfoFlags & rdStallConditionMask);
}

/* ============================================================================
 *  IsLoadUseStall: Determines a scalar load/use condition is present.
 * ========================================================================= */
static bool
IsLoadUseStall(const struct RSPRDEXLatch *rdexLatch, uint32_t exInfoFlags,
  uint32_t dfInfoFlags, uint32_t exDestination, uint32_t dfDestination,
  unsigned rsSource, unsigned rtSource) {
  uint32_t rdInfoFlags = rdexLatch->opcode.infoFlags;
  bool rdRequiresRS = rdInfoFlags & OPCODE_INFO_NEED_RS;
  bool rdRequiresRT = rdInfoFlags & OPCODE_INFO_NEED_RT;

  if ((exInfoFlags & OPCODE_INFO_LOAD) && 
    ((rdRequiresRS && rsSource == exDestination) ||
    (rdRequiresRT && rtSource == exDestination)))
    return true;

  return ((dfInfoFlags & OPCODE_INFO_LOAD) &&
    ((rdRequiresRS && rsSource == dfDestination) ||
    (rdRequiresRT && rtSource == dfDestination)));
}

/* ============================================================================
 *  CycleRSP: Advances the state of the processor pipeline one cycle.
 * ========================================================================= */
void
CycleRSP(struct RSP *rsp) {
  bool ldStoreStall, ldUseStall;
  unsigned rsSource = GET_RS(rsp->pipeline.rdexLatch.iw);
  unsigned rtSource = GET_RT(rsp->pipeline.rdexLatch.iw);

  /* Generate outputs for later stages. */ 
  struct RSPOpcode dfOpcode = rsp->pipeline.exdfLatch.opcode;
  struct RSPOpcode exOpcode = rsp->pipeline.rdexLatch.opcode;
  struct RSPOpcode rfOpcode;

  /* If we're halted, just bail out. */
  if (rsp->cp0.regs[SP_STATUS_REG] & 0x1)
    return;

  RSPWBStage(rsp);
  RSPDFStage(rsp);

  /* Execute and bump opcode counters. */
  RSPEXStage(rsp, rsSource, rtSource);
  RSPCycleCP2(&rsp->cp2);
  RSPRDStage(rsp);

#ifndef NDEBUG
  rsp->pipeline.cycles++;
#endif

  /* Check for stall conditions. */
  rfOpcode = rsp->pipeline.rdexLatch.opcode;

  ldStoreStall = IsLoadStoreStall(rfOpcode.infoFlags, dfOpcode.infoFlags);
  ldUseStall = IsLoadUseStall(&rsp->pipeline.rdexLatch, exOpcode.infoFlags,
    exOpcode.infoFlags, rsp->pipeline.exdfLatch.result.dest,
    rsp->pipeline.dfwbLatch.result.dest, rsSource, rtSource);

  /* Fetch if there were no stalls. */
  if (unlikely(ldStoreStall | ldUseStall)) {
    RSPInvalidateOpcode(&rsp->pipeline.rdexLatch.opcode);
    RSPInvalidateVectorOpcode(&rsp->cp2.opcode);
  }

  else
    RSPIFStage(rsp);
}

/* ============================================================================
 *  RSPInitPipeline: Initializes the pipeline.
 * ========================================================================= */
void
RSPInitPipeline(struct RSPPipeline *pipeline) {
  RSPInvalidateOpcode(&pipeline->rdexLatch.opcode);
  RSPInvalidateOpcode(&pipeline->exdfLatch.opcode);

  pipeline->ifrdLatch.firstIW = 0;
  pipeline->ifrdLatch.secondIW = 0;
  pipeline->ifrdLatch.pc = 0x1000;
}

