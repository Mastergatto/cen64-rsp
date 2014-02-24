/* ============================================================================
 *  CP0.c: RSP Coprocessor #0.
 *
 *  RSPSIM: Reality Signal Processor SIMulator.
 *  Copyright (C) 2013, Tyler J. Stachecki.
 *  All rights reserved.
 *
 *  This file is subject to the terms and conditions defined in
 *  file 'LICENSE', which is part of this source code package.
 * ========================================================================= */
#include "Address.h"
#include "Common.h"
#include "CP0.h"
#include "CPU.h"
#include "Decoder.h"
#include "Interface.h"

#ifdef __cplusplus
#include <cassert>
#include <cstring>
#else
#include <assert.h>
#include <string.h>
#endif

/* ============================================================================
 *  Mnemonics table.
 * ========================================================================= */
#ifndef NDEBUG
const char *SPRegisterMnemonics[NUM_SP_REGISTERS] = {
#define X(reg) #reg,
#include "Registers.md"
#undef X
};
#endif

/* ============================================================================
 *  Instruction: MFC0 (Move From System Control Coprocessor)
 * ========================================================================= */
void
RSPMFC0(struct RSP *rsp, uint32_t unused(rs), uint32_t unused(rt)) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;

  unsigned rd = GET_RD(rdexLatch->iw);
  unsigned dest = GET_RT(rdexLatch->iw);
  unsigned result;

  (rd >= CMD_START)
    ? DPRegRead(rsp->rdp, DP_REGS_BASE_ADDRESS + ((rd - CMD_START) << 2), &result)
    : SPRegRead(rsp, SP_REGS_BASE_ADDRESS + (rd << 2), &result);

  exdfLatch->result.data = result;
  exdfLatch->result.dest = dest;
}

/* ============================================================================
 *  Instruction: MTC0 (Move To System Control Coprocessor)
 * ========================================================================= */
void
RSPMTC0(struct RSP *rsp, uint32_t unused(rs), uint32_t rt) {
  const struct RSPRDEXLatch *rdexLatch = &rsp->pipeline.rdexLatch;
  struct RSPEXDFLatch *exdfLatch = &rsp->pipeline.exdfLatch;
  unsigned rd = rdexLatch->iw >> 11 & 0x1F;

  (rd >= CMD_START)
    ? DPRegWrite(rsp->rdp, DP_REGS_BASE_ADDRESS + ((rd - CMD_START) << 2), &rt)
    : SPRegWrite(rsp, SP_REGS_BASE_ADDRESS + (rd << 2), &rt);

  memset(&exdfLatch->result, 0, sizeof(exdfLatch->result));
}

/* ============================================================================
 *  RSPInitCP0: Initializes the co-processor.
 * ========================================================================= */
void
RSPInitCP0(struct RSPCP0 *cp) {
  debug("Initializing CP0.");
  memset(cp, 0, sizeof(*cp));

  cp->regs[SP_STATUS_REG] = 0x1;
}

