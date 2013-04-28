/* ============================================================================
 *  Interface.c: Reality Shader Processor (RSP) Interface.
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
#include "CPU.h"
#include "Definitions.h"
#include "Interface.h"

#ifdef __cplusplus
#include <cassert>
#include <cstring>
#else
#include <assert.h>
#include <string.h>
#endif

static void HandleDMARead(struct RSP *);
static void HandleDMAWrite(struct RSP *);
static void HandleSPStatusWrite(struct RSP *, uint32_t);

/* ============================================================================
 *  HandleDMARead: Invoked when SP_RD_LEN_REG is written.
 *
 *  SP_DRAM_ADDR_REG = RDRAM (source) address.
 *  SP_MEM_ADDR_REG = I/DCache (target) address.
 *  SP_RD_LEN_REG =  Skip | Count | Transfer size.
 * ========================================================================= */
static void
HandleDMARead(struct RSP *rsp) {
  uint32_t dest = rsp->cp0.regs[SP_MEM_ADDR_REG];
  uint32_t source = rsp->cp0.regs[SP_DRAM_ADDR_REG];
  uint32_t length = (rsp->cp0.regs[SP_RD_LEN_REG] & 0xFFF) + 1;
  uint32_t skip = rsp->cp0.regs[SP_RD_LEN_REG] >> 20 & 0xFFF;

  unsigned count = rsp->cp0.regs[SP_RD_LEN_REG] >> 12 & 0xFF;
  unsigned i;

  /* Check alignment. */
  if (length & 0x7)
    length = (length + 0x7) & ~0x7;

  if (source & 0x3)
    source &= ~0x3;

  if (dest & 0x7)
    dest &= ~0x7;

  for (i = 0; i <= count; i++) {
    debug("DMA | Request: Read from DRAM.");
    debugarg("DMA | DEST   : [0x%.8x].", dest);
    debugarg("DMA | SOURCE : [0x%.8x].", source);
    debugarg("DMA | LENGTH : [0x%.8x].", length);

    DMAFromDRAM(rsp->bus, rsp->dmem + dest, source, length);

    source += length + skip;
    dest += length;
  }

  /* Update the registers. */
  rsp->cp0.regs[SP_DRAM_ADDR_REG] = source;
  rsp->cp0.regs[SP_MEM_ADDR_REG] = dest;
}

/* ============================================================================
 *  HandleDMAWrite: Invoked when SP_WR_LEN_REG is written.
 *
 *  SP_MEM_ADDR_REG = I/DCache (source) address.
 *  SP_DRAM_ADDR_REG = RDRAM (target) address.
 *  SP_RD_LEN_REG =  Skip | Count | Transfer size.
 * ========================================================================= */
static void
HandleDMAWrite(struct RSP *rsp) {
  uint32_t dest = rsp->cp0.regs[SP_DRAM_ADDR_REG];
  uint32_t source = rsp->cp0.regs[SP_MEM_ADDR_REG];
  uint32_t length = (rsp->cp0.regs[SP_WR_LEN_REG] & 0xFFF) + 1;
  uint32_t skip = rsp->cp0.regs[SP_WR_LEN_REG] >> 20 & 0xFFF;

  unsigned count = rsp->cp0.regs[SP_WR_LEN_REG] >> 12 & 0xFF;
  unsigned i;

  /* Check alignment. */
  if (length & 0x7)
    length = (length + 0x7) & ~0x7;

  if (dest & 0x3)
    dest &= ~0x3;

  if (source & 0x7)
    source &= ~0x7;

  for (i = 0; i <= count; i++) {
    debug("DMA | Request: Write to DRAM.");
    debugarg("DMA | DEST   : [0x%.8x].", dest);
    debugarg("DMA | SOURCE : [0x%.8x].", source);
    debugarg("DMA | LENGTH : [0x%.8x].", length);

    DMAToDRAM(rsp->bus, dest, rsp->dmem + source, length);

    source += length;
    dest += length + skip;
  }

  /* Update the registers. */
  rsp->cp0.regs[SP_MEM_ADDR_REG] += source;
  rsp->cp0.regs[SP_DRAM_ADDR_REG] += dest;
}

/* ============================================================================
 *  HandleSPStatusWrite: Update state after a write to SP_STATUS_REG.
 * ========================================================================= */
static void
HandleSPStatusWrite(struct RSP *rsp, uint32_t data) {
  assert(!((data & SP_CLR_HALT) && (data & SP_SET_HALT)));
  assert(!((data & SP_CLR_INTR) && (data & SP_SET_INTR)));
  assert(!((data & SP_CLR_SSTEP) && (data & SP_SET_SSTEP)));
  assert(!((data & SP_CLR_INTR_BREAK) && (data & SP_SET_INTR_BREAK)));

  assert(!((data & SP_CLR_SIG0) && (data & SP_SET_SIG0)));
  assert(!((data & SP_CLR_SIG1) && (data & SP_SET_SIG1)));
  assert(!((data & SP_CLR_SIG2) && (data & SP_SET_SIG2)));
  assert(!((data & SP_CLR_SIG3) && (data & SP_SET_SIG3)));
  assert(!((data & SP_CLR_SIG4) && (data & SP_SET_SIG4)));
  assert(!((data & SP_CLR_SIG5) && (data & SP_SET_SIG5)));
  assert(!((data & SP_CLR_SIG6) && (data & SP_SET_SIG6)));
  assert(!((data & SP_CLR_SIG7) && (data & SP_SET_SIG7)));

  if (data & SP_CLR_HALT)
    rsp->cp0.regs[SP_STATUS_REG] &= ~SP_STATUS_HALT;
  else if (data & SP_SET_HALT)
    rsp->cp0.regs[SP_STATUS_REG] |= SP_STATUS_HALT;

  if (data & SP_CLR_BROKE)
    rsp->cp0.regs[SP_STATUS_REG] &= ~SP_STATUS_BROKE;

  if (data & SP_CLR_INTR)
    BusClearRCPInterrupt(rsp->bus, MI_INTR_SP);
  else if (data & SP_SET_INTR)
    BusRaiseRCPInterrupt(rsp->bus, MI_INTR_SP);

  if (data & SP_CLR_SSTEP)
    rsp->cp0.regs[SP_STATUS_REG] &= ~SP_STATUS_SSTEP;
  else if (data & SP_SET_SSTEP)
    rsp->cp0.regs[SP_STATUS_REG] |= SP_STATUS_SSTEP;

  if (data & SP_CLR_INTR_BREAK)
    rsp->cp0.regs[SP_STATUS_REG] &= ~SP_STATUS_INTR_BREAK;
  else if  (data & SP_SET_INTR_BREAK)
    rsp->cp0.regs[SP_STATUS_REG] |= SP_STATUS_INTR_BREAK;

  if (data & SP_CLR_SIG0)
    rsp->cp0.regs[SP_STATUS_REG] &= ~SP_STATUS_SIG0;
  else if (data & SP_SET_SIG0)
    rsp->cp0.regs[SP_STATUS_REG] |= SP_STATUS_SIG0;

  if (data & SP_CLR_SIG1)
    rsp->cp0.regs[SP_STATUS_REG] &= ~SP_STATUS_SIG1;
  else if (data & SP_SET_SIG1)
    rsp->cp0.regs[SP_STATUS_REG] |= SP_STATUS_SIG1;

  if (data & SP_CLR_SIG2)
    rsp->cp0.regs[SP_STATUS_REG] &= ~SP_STATUS_SIG2;
  else if (data & SP_SET_SIG2)
    rsp->cp0.regs[SP_STATUS_REG] |= SP_STATUS_SIG2;

  if (data & SP_CLR_SIG3)
    rsp->cp0.regs[SP_STATUS_REG] &= ~SP_STATUS_SIG3;
  else if (data & SP_SET_SIG3)
    rsp->cp0.regs[SP_STATUS_REG] |= SP_STATUS_SIG3;

  if (data & SP_CLR_SIG4)
    rsp->cp0.regs[SP_STATUS_REG] &= ~SP_STATUS_SIG4;
  else if (data & SP_SET_SIG4)
    rsp->cp0.regs[SP_STATUS_REG] |= SP_STATUS_SIG4;

  if (data & SP_CLR_SIG5)
    rsp->cp0.regs[SP_STATUS_REG] &= ~SP_STATUS_SIG5;
  else if (data & SP_SET_SIG5)
    rsp->cp0.regs[SP_STATUS_REG] |= SP_STATUS_SIG5;

  if (data & SP_CLR_SIG6)
    rsp->cp0.regs[SP_STATUS_REG] &= ~SP_STATUS_SIG6;
  else if (data & SP_SET_SIG6)
    rsp->cp0.regs[SP_STATUS_REG] |= SP_STATUS_SIG6;

  if (data & SP_CLR_SIG7)
    rsp->cp0.regs[SP_STATUS_REG] &= ~SP_STATUS_SIG7;
  else if (data & SP_SET_SIG7)
    rsp->cp0.regs[SP_STATUS_REG] |= SP_STATUS_SIG7;
}

/* ============================================================================
 *  RSPDMemReadWord: Read a word from RSP's DMEM.
 * ========================================================================= */
int RSPDMemReadWord(void *_rsp, uint32_t address, void *_data) {
	struct RSP *rsp = (struct RSP*) _rsp;
	uint32_t *data = (uint32_t*) _data, word;

  address = address - RSP_DMEM_BASE_ADDRESS;

  memcpy(&word, rsp->dmem + address, sizeof(word));
  *data = ByteOrderSwap32(word);

  return 0;
}

/* ============================================================================
 *  RSPDMemWriteWord: Write a word to RSP's DMEM.
 * ========================================================================= */
int RSPDMemWriteWord(void *_rsp, uint32_t address, void *_data) {
	struct RSP *rsp = (struct RSP*) _rsp;
	uint32_t *data = (uint32_t*) _data, word;
 
  address = address - RSP_DMEM_BASE_ADDRESS;
  word = ByteOrderSwap32(*data);

  memcpy(rsp->dmem + address, &word, sizeof(word));

  return 0;
}

/* ============================================================================
 *  RSPIMemReadByte: Read a byte from RSP's IMEM.
 * ========================================================================= */
int RSPIMemReadByte(void *_rsp, uint32_t address, void *_data) {
	struct RSP *rsp = (struct RSP*) _rsp;
  uint8_t *data = (uint8_t *) _data, byte;

  address = address - RSP_IMEM_BASE_ADDRESS;

  memcpy(&byte, rsp->imem + address, sizeof(byte));
  *data = byte;

  return 0;
}


/* ============================================================================
 *  RSPIMemReadWord: Read a word from RSP's IMEM.
 * ========================================================================= */
int RSPIMemReadWord(void *_rsp, uint32_t address, void *_data) {
	struct RSP *rsp = (struct RSP*) _rsp;
  uint32_t *data = (uint32_t *) _data, word;

  address = address - RSP_IMEM_BASE_ADDRESS;

  memcpy(&word, rsp->imem + address, sizeof(word));
  *data = ByteOrderSwap32(word);

  return 0;
}

/* ============================================================================
 *  RSPIMemWriteByte: Write a byte to RSP's IMEM.
 * ========================================================================= */
int RSPIMemWriteByte(void *_rsp, uint32_t address, void *_data) {
	struct RSP *rsp = (struct RSP*) _rsp;
	uint8_t *data = (uint8_t*) _data, byte;
 
  address = address - RSP_IMEM_BASE_ADDRESS;
  byte = *data;

  memcpy(rsp->imem + address, &byte, sizeof(byte));

  return 0;
}

/* ============================================================================
 *  RSPIMemWriteWord: Write a word to RSP's IMEM.
 * ========================================================================= */
int RSPIMemWriteWord(void *_rsp, uint32_t address, void *_data) {
	struct RSP *rsp = (struct RSP*) _rsp;
	uint32_t *data = (uint32_t*) _data, word;
 
  address = address - RSP_IMEM_BASE_ADDRESS;
  word = ByteOrderSwap32(*data);

  memcpy(rsp->imem + address, &word, sizeof(word));

  return 0;
}

/* ============================================================================
 *  SPRegRead: Read from SP registers.
 * ========================================================================= */
int
SPRegRead(void *_rsp, uint32_t address, void *_data) {
	struct RSP *rsp = (struct RSP*) _rsp;
	uint32_t *data = (uint32_t*) _data;

  address -= SP_REGS_BASE_ADDRESS;
  enum SPRegister reg = (enum SPRegister) (address / 4);

  debugarg("SPRegRead: Reading from register [%s].", SPRegisterMnemonics[reg]);

  *data = rsp->cp0.regs[reg];
  if (reg == SP_SEMAPHORE_REG)
    rsp->cp0.regs[SP_SEMAPHORE_REG] = 1;

  return 0;
}

/* ============================================================================
 *  SPRegRead2: Read from second set of SP registers [PC, BIST].
 * ========================================================================= */
int
SPRegRead2(void *_rsp, uint32_t address, void *_data) {
	struct RSP *rsp = (struct RSP*) _rsp;
	uint32_t *data = (uint32_t*) _data;

  address -= SP_REGS2_BASE_ADDRESS;
  enum SPRegister reg = (enum SPRegister) ((address / 4) + SP_PC_REG);

  debugarg("SPRegRead: Reading from register [%s].", SPRegisterMnemonics[reg]);

  *data = rsp->cp0.regs[reg];
  return 0;
}

/* ============================================================================
 *  SPRegWrite: Write to SP registers.
 * ========================================================================= */
int
SPRegWrite(void *_rsp, uint32_t address, void *_data) {
	struct RSP *rsp = (struct RSP*) _rsp;
	uint32_t *data = (uint32_t*) _data;

  address -= SP_REGS_BASE_ADDRESS;
  enum SPRegister reg = (enum SPRegister) (address / 4);

  debugarg("SPRegWrite: Writing to register [%s].", SPRegisterMnemonics[reg]);

  switch(reg) {
  case SP_MEM_ADDR_REG:
    rsp->cp0.regs[SP_MEM_ADDR_REG] = *data & 0x1FFF;
    break;

  case SP_DRAM_ADDR_REG:
    rsp->cp0.regs[reg] = *data & 0xFFFFFF;
    break;

  case SP_RD_LEN_REG:
    rsp->cp0.regs[SP_RD_LEN_REG] = *data;
    HandleDMARead(rsp);
    break;

  case SP_WR_LEN_REG:
    rsp->cp0.regs[SP_WR_LEN_REG] = *data;
    HandleDMAWrite(rsp);
    break;

  case SP_STATUS_REG:
    HandleSPStatusWrite(rsp, *data);
    break;

  default:
    rsp->cp0.regs[reg] = *data;
    break;
  }

  return 0;
}

/* ============================================================================
 *  SPRegWrite2: Write to second set of SP registers [PC, BIST].
 * ========================================================================= */
int
SPRegWrite2(void *_rsp, uint32_t address, void *_data) {
	struct RSP *rsp = (struct RSP*) _rsp;
	uint32_t *data = (uint32_t*) _data;

  address -= SP_REGS2_BASE_ADDRESS;
  enum SPRegister reg = (enum SPRegister) ((address / 4) + SP_PC_REG);

  debugarg("SPRegWrite: Writing to register [%s].", SPRegisterMnemonics[reg]);

  if (reg == SP_PC_REG) {
    RSPInitPipeline(&rsp->pipeline); /* Hack? */
    rsp->pipeline.ifrdLatch.pc = *data & 0xFFF;
  }

  rsp->cp0.regs[reg] = *data;
  return 0;
}
