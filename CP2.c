/* ============================================================================
 *  CP2.c: RSP Coprocessor #2.
 *
 *  RSPSIM: Reality Signal Processor SIMulator.
 *  Copyright (C) 2013, Tyler J. Stachecki.
 *  All rights reserved.
 *
 *  This file is subject to the terms and conditions defined in
 *  file 'LICENSE', which is part of this source code package.
 * ========================================================================= */
#include "Common.h"
#include "CP2.h"
#include "Decoder.h"
#include "Opcodes.h"
#include "ReciprocalROM.h"

#ifdef __cplusplus
#include <cassert>
#include <cstdio>
#include <cstring>
#else
#include <assert.h>
#include <stdio.h>
#include <string.h>
#endif

#ifdef USE_SSE
#ifdef SSSE3_ONLY
#include <tmmintrin.h>
#else
#include <smmintrin.h>
#endif

/* ============================================================================
 *  RSPClampLowToVal: Clamps the low word of the accumulator.
 * ========================================================================= */
static __m128i
RSPClampLowToVal(__m128i vaccLow,
  __m128i vaccMid, __m128i vaccHigh, __m128i zero) {
  __m128i midSign, negCheck, negVal, posVal, useValMask;

  /* Compute some common values ahead of time. */
  negCheck = _mm_cmplt_epi16(vaccHigh, zero);
  midSign = _mm_srai_epi16(vaccMid, 15);

  /* If accmulator < 0, clamp to val if val != TMin. */
  negVal = _mm_and_si128(negCheck, vaccHigh);
  useValMask = _mm_and_si128(midSign, negVal);
  negVal = _mm_and_si128(vaccLow, useValMask);

  /* Otherwise, clamp to ~0 if any high bits are set. */
  useValMask = _mm_or_si128(vaccHigh,  midSign);
  useValMask = _mm_cmpeq_epi16(useValMask, zero);
  posVal = _mm_and_si128(useValMask, vaccLow);

#ifdef SSSE3_ONLY
  negVal = _mm_and_si128(negCheck, negVal);
  posVal = _mm_andnot_si128(negCheck, posVal);
  return _mm_or_si128(negVal, posVal);
#else
  return _mm_blendv_epi8(posVal, negVal, negCheck);
#endif
}

/* ============================================================================
 *  RSPGetVCC: Get VCC in the "old" format.
 * ========================================================================= */
#ifdef USE_SSE
uint16_t
RSPGetVCC(const struct RSPCP2 *cp2) {
  __m128i vge = _mm_load_si128((__m128i*) (cp2->vcchi.slices));
  __m128i vle = _mm_load_si128((__m128i*) (cp2->vcclo.slices));
  return _mm_movemask_epi8(_mm_packs_epi16(vle, vge));
}
#endif

/* ============================================================================
 *  RSPGetVCE: Get VCE in the "old" format.
 * ========================================================================= */
#ifdef USE_SSE
uint8_t
RSPGetVCE(const struct RSPCP2 *cp2) {
  __m128i vce = _mm_load_si128((__m128i*) (cp2->vce.slices));
  return _mm_movemask_epi8(_mm_packs_epi16(vce, vce));
}
#endif

/* ============================================================================
 *  RSPGetVCO: Get VCO in the "old" format.
 * ========================================================================= */
#ifdef USE_SSE
uint16_t
RSPGetVCO(const struct RSPCP2 *cp2) {
  __m128i vne = _mm_load_si128((__m128i*) (cp2->vcohi.slices));
  __m128i vco = _mm_load_si128((__m128i*) (cp2->vcolo.slices));
  return _mm_movemask_epi8(_mm_packs_epi16(vco, vne));
}
#endif

/* ============================================================================
 *  RSPGetVectorOperands: Builds and returns the proper configuration of the
 *  `vt` vector for instructions that require the use of a element specifier.
 * ========================================================================= */
static __m128i
RSPGetVectorOperands(__m128i vt, unsigned element) {
  static const uint8_t VectorOperandsArray[16][16] align(16) = {
    /* pshufb (_mm_shuffle_epi8) keys. */
    /* -- */ {0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,0xA,0xB,0xC,0xD,0xE,0xF},
    /* -- */ {0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,0xA,0xB,0xC,0xD,0xE,0xF},
    /* 0q */ {0x0,0x1,0x0,0x1,0x4,0x5,0x4,0x5,0x8,0x9,0x8,0x9,0xC,0xD,0xC,0xD},
    /* 1q */ {0x2,0x3,0x2,0x3,0x6,0x7,0x6,0x7,0xA,0xB,0xA,0xB,0xE,0xF,0xE,0xF},
    /* 0h */ {0x0,0x1,0x0,0x1,0x0,0x1,0x0,0x1,0x8,0x9,0x8,0x9,0x8,0x9,0x8,0x9},
    /* 1h */ {0x2,0x3,0x2,0x3,0x2,0x3,0x2,0x3,0xA,0xB,0xA,0xB,0xA,0xB,0xA,0xB},
    /* 2h */ {0x4,0x5,0x4,0x5,0x4,0x5,0x4,0x5,0xC,0xD,0xC,0xD,0xC,0xD,0xC,0xD},
    /* 3h */ {0x6,0x7,0x6,0x7,0x6,0x7,0x6,0x7,0xE,0xF,0xE,0xF,0xE,0xF,0xE,0xF},
    /* 0w */ {0x0,0x1,0x0,0x1,0x0,0x1,0x0,0x1,0x0,0x1,0x0,0x1,0x0,0x1,0x0,0x1},
    /* 1w */ {0x2,0x3,0x2,0x3,0x2,0x3,0x2,0x3,0x2,0x3,0x2,0x3,0x2,0x3,0x2,0x3},
    /* 2w */ {0x4,0x5,0x4,0x5,0x4,0x5,0x4,0x5,0x4,0x5,0x4,0x5,0x4,0x5,0x4,0x5},
    /* 3w */ {0x6,0x7,0x6,0x7,0x6,0x7,0x6,0x7,0x6,0x7,0x6,0x7,0x6,0x7,0x6,0x7},
    /* 4w */ {0x8,0x9,0x8,0x9,0x8,0x9,0x8,0x9,0x8,0x9,0x8,0x9,0x8,0x9,0x8,0x9},
    /* 5w */ {0xA,0xB,0xA,0xB,0xA,0xB,0xA,0xB,0xA,0xB,0xA,0xB,0xA,0xB,0xA,0xB},
    /* 6w */ {0xC,0xD,0xC,0xD,0xC,0xD,0xC,0xD,0xC,0xD,0xC,0xD,0xC,0xD,0xC,0xD},
    /* 7w */ {0xE,0xF,0xE,0xF,0xE,0xF,0xE,0xF,0xE,0xF,0xE,0xF,0xE,0xF,0xE,0xF}
  };

  __m128i key = _mm_load_si128((__m128i*) VectorOperandsArray[element]);
  return _mm_shuffle_epi8(vt, key);
}

/* ============================================================================
 *  RSPPackLo32to16: Pack LSBs of 32-bit vectors to 16-bits without saturation.
 *  TODO: 5 SSE2 operations is kind of expensive just to truncate values?
 * ========================================================================= */
static __m128i
RSPPackLo32to16(__m128i vectorLow, __m128i vectorHigh, __m128i zero) {
#ifdef SSSE3_ONLY
  vectorLow = _mm_slli_epi32(vectorLow, 16);
  vectorHigh = _mm_slli_epi32(vectorHigh, 16);
  vectorLow = _mm_srai_epi32(vectorLow, 16);
  vectorHigh = _mm_srai_epi32(vectorHigh, 16);
  return _mm_packs_epi32(vectorLow, vectorHigh);
#else
  vectorLow = _mm_blend_epi16(vectorLow, zero, 0xAA);
  vectorHigh = _mm_blend_epi16(vectorHigh, zero, 0xAA);
  return _mm_packus_epi32(vectorLow, vectorHigh);
#endif
}

/* ============================================================================
 *  RSPPackHi32to16: Pack MSBs of 32-bit vectors to 16-bits without saturation.
 * ========================================================================= */
static __m128i
RSPPackHi32to16(__m128i vectorLow, __m128i vectorHigh) {
  vectorLow = _mm_srai_epi32(vectorLow, 16);
  vectorHigh = _mm_srai_epi32(vectorHigh, 16);
  return _mm_packs_epi32(vectorLow, vectorHigh);
}

/* ============================================================================
 *  RSPPack32to16: Packs 2x32-bit vectors to two parallel 16-bit vectors.
 * ========================================================================= */
static void
RSPPack32to16(__m128i low, __m128i high, __m128i *lowOut, __m128i *highOut) {
  __m128i key = _mm_setr_epi8(
    0x0,0x1,0x4,0x5,0x8,0x9,0xC,0xD,
    0x2,0x3,0x6,0x7,0xA,0xB,0xE,0xF
  );

  low = _mm_shuffle_epi8(low, key);
  high = _mm_shuffle_epi8(high, key);
  *lowOut = _mm_unpacklo_epi64(low, high);
  *highOut = _mm_unpackhi_epi64(low, high);
}

#ifdef USE_SSE
static const uint16_t setLUT[16][4] align(64) = {
  {0x0000U, 0x0000U, 0x0000U, 0x0000},
  {0xFFFFU, 0x0000U, 0x0000U, 0x0000},
  {0x0000U, 0xFFFFU, 0x0000U, 0x0000},
  {0xFFFFU, 0xFFFFU, 0x0000U, 0x0000},
  {0x0000U, 0x0000U, 0xFFFFU, 0x0000},
  {0xFFFFU, 0x0000U, 0xFFFFU, 0x0000},
  {0x0000U, 0xFFFFU, 0xFFFFU, 0x0000},
  {0xFFFFU, 0xFFFFU, 0xFFFFU, 0x0000},
  {0x0000U, 0x0000U, 0x0000U, 0xFFFF},
  {0xFFFFU, 0x0000U, 0x0000U, 0xFFFF},
  {0x0000U, 0xFFFFU, 0x0000U, 0xFFFF},
  {0xFFFFU, 0xFFFFU, 0x0000U, 0xFFFF},
  {0x0000U, 0x0000U, 0xFFFFU, 0xFFFF},
  {0xFFFFU, 0x0000U, 0xFFFFU, 0xFFFF},
  {0x0000U, 0xFFFFU, 0xFFFFU, 0xFFFF},
  {0xFFFFU, 0xFFFFU, 0xFFFFU, 0xFFFF},
};

/* ============================================================================
 *  RSPSetVCC: Set VCC given the "old" format.
 * ========================================================================= */
void
RSPSetVCC(struct RSPCP2 *cp2, uint16_t vcc) {
  memcpy(cp2->vcclo.slices + 0, setLUT[(vcc >>  0) & 0xF], sizeof(*setLUT));
  memcpy(cp2->vcclo.slices + 4, setLUT[(vcc >>  4) & 0xF], sizeof(*setLUT));
  memcpy(cp2->vcchi.slices + 0, setLUT[(vcc >>  8) & 0xF], sizeof(*setLUT));
  memcpy(cp2->vcchi.slices + 4, setLUT[(vcc >> 12) & 0xF], sizeof(*setLUT));
}

/* ============================================================================
 *  RSPSetVCE: Set VCE given the "old" format.
 * ========================================================================= */
void
RSPSetVCE(struct RSPCP2 *cp2, uint8_t vce) {
  memcpy(cp2->vce.slices + 0, setLUT[(vce >>  0) & 0xF], sizeof(*setLUT));
  memcpy(cp2->vce.slices + 4, setLUT[(vce >>  4) & 0xF], sizeof(*setLUT));
}

/* ============================================================================
 *  RSPSetVCO: Set VCO given the "old" format.
 * ========================================================================= */
void
RSPSetVCO(struct RSPCP2 *cp2, uint16_t vco) {
  memcpy(cp2->vcolo.slices + 0, setLUT[(vco >>  0) & 0xF], sizeof(*setLUT));
  memcpy(cp2->vcolo.slices + 4, setLUT[(vco >>  4) & 0xF], sizeof(*setLUT));
  memcpy(cp2->vcohi.slices + 0, setLUT[(vco >>  8) & 0xF], sizeof(*setLUT));
  memcpy(cp2->vcohi.slices + 4, setLUT[(vco >> 12) & 0xF], sizeof(*setLUT));
}
#endif

/* ============================================================================
 *  RSPSignExtend16to32: Sign-extend 16-bit slices to 32-bit slices.
 * ========================================================================= */
static void
RSPSignExtend16to32(__m128i source, __m128i *vectorLow, __m128i *vectorHigh) {
  __m128i vMask = _mm_srai_epi16(source, 15);
  *vectorHigh = _mm_unpackhi_epi16(source, vMask);
  *vectorLow = _mm_unpacklo_epi16(source, vMask);
}

/* ============================================================================
 *  RSPZeroExtend16to32: Zero-extend 16-bit slices to 32-bit slices.
 * ========================================================================= */
static void
RSPZeroExtend16to32(__m128i source,
  __m128i *vectorLow, __m128i *vectorHigh, __m128i zero) {
  *vectorHigh = _mm_unpackhi_epi16(source, zero);
  *vectorLow = _mm_unpacklo_epi16(source, zero);
}

/* ============================================================================
 *  SSE lacks nand, nor, and nxor (really, xnor), so define them manually.
 * ========================================================================= */
static __m128i
_mm_nand_si128(__m128i a, __m128i b) {
  __m128i mask = _mm_cmpeq_epi8(a, a);
  return _mm_xor_si128(_mm_and_si128(a, b), mask);
}

static __m128i
_mm_nor_si128(__m128i a, __m128i b) {
  __m128i mask = _mm_cmpeq_epi8(a, a);
  return _mm_xor_si128(_mm_or_si128(a, b), mask);
}

static __m128i
_mm_nxor_si128(__m128i a, __m128i b) {
  __m128i mask = _mm_cmpeq_epi8(a, a);
  return _mm_xor_si128(_mm_xor_si128(a, b), mask);
}

/* ============================================================================
 *  _mm_mullo_epi32: SSE2 lacks _mm_mullo_epi32, define it manually.
 *  TODO/WARNING/DISCLAIMER: Assumes one argument is positive.
 * ========================================================================= */
#ifdef SSSE3_ONLY
static __m128i
_mm_mullo_epi32(__m128i a, __m128i b) {
  __m128i a4 = _mm_srli_si128(a, 4);
  __m128i b4 = _mm_srli_si128(b, 4);
  __m128i ba = _mm_mul_epu32(b, a);
  __m128i b4a4 = _mm_mul_epu32(b4, a4);

  __m128i mask = _mm_setr_epi32(~0, 0, ~0, 0);
  __m128i baMask = _mm_and_si128(ba, mask);
  __m128i b4a4Mask = _mm_and_si128(b4a4, mask);
  __m128i b4a4MaskShift = _mm_slli_si128(b4a4Mask, 4);

  return _mm_or_si128(baMask, b4a4MaskShift);
}
#endif
#endif

/* ============================================================================
 *  Instruction: VABS (Vector Absolute Value of Short Elements)
 * ========================================================================= */
__m128i
RSPVABS(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i valLessThan, signLessThan, resultLessThan, vdReg;
  vdReg = _mm_sign_epi16(vtShuf, vsReg);

  /* _mm_sign_epi16 will not fixup INT16_MIN; the RSP will! */
  resultLessThan = _mm_cmplt_epi16(vdReg, zero);
  signLessThan = _mm_cmplt_epi16(vsReg, zero);
  valLessThan = _mm_cmplt_epi16(vtShuf, zero);

  valLessThan = _mm_and_si128(valLessThan, signLessThan);
  resultLessThan = _mm_and_si128(valLessThan, resultLessThan);
  vdReg = _mm_xor_si128(vdReg, resultLessThan);

  _mm_store_si128((__m128i*) accLow, vdReg);
  return vdReg;
#else
#warning "Unimplemented function: RSPVABS (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VADD (Vector Add of Short Elements)
 * ========================================================================= */
__m128i
RSPVADD(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i minimum, maximum, carryOut;
  __m128i vdReg, vaccLow;

  carryOut = _mm_load_si128((__m128i*) (cp2->vcolo.slices));
  carryOut = _mm_srli_epi16(carryOut, 15);

  /* VACC uses unsaturated arithmetic. */
  vdReg = _mm_add_epi16(vsReg, vtShuf);
  vaccLow = _mm_add_epi16(vdReg, carryOut);

  /* VD is the signed sum of the two sources and the carry. Since we */
  /* have to saturate the sum of all three, we have to be clever. */
  minimum = _mm_min_epi16(vsReg, vtShuf);
  maximum = _mm_max_epi16(vsReg, vtShuf);
  minimum = _mm_adds_epi16(minimum, carryOut);
  vdReg = _mm_adds_epi16(minimum, maximum);

  _mm_store_si128((__m128i*) accLow, vaccLow);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), zero);
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), zero);
  return vdReg;
#else
#warning "Unimplemented function: RSPVADD (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VADDC (Vector Add of Short Elements with Carry)
 * ========================================================================= */
__m128i
RSPVADDC(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i satSum, unsatSum, equalMask;
  satSum = _mm_adds_epu16(vsReg, vtShuf);
  unsatSum = _mm_add_epi16(vsReg, vtShuf);

  equalMask = _mm_cmpeq_epi16(satSum, unsatSum);
  equalMask = _mm_cmpeq_epi16(equalMask, zero);

  _mm_store_si128((__m128i*) accLow, unsatSum);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), equalMask);
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), zero);
  return unsatSum;
#else
#warning "Unimplemented function: RSPVADDC (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VAND (Vector AND of Short Elements)
 * ========================================================================= */
__m128i
RSPVAND(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i unused(zero)) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vdReg;

  vdReg = _mm_and_si128(vtShuf, vsReg);
  _mm_store_si128((__m128i*) accLow, vdReg);
  return vdReg;
#else
#warning "Unimplemented function: RSPVAND (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VCH (Vector Select Clip Test High)
 * ========================================================================= */
__m128i
RSPVCH(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i cmp1, cmp2, negVtReg, snAluOp, temp, temp2a, temp2s;
  __m128i ge, le, neq, sn;

  /* sn = (vs ^ vt) < 0 */
  sn = _mm_xor_si128(vsReg, vtShuf);
  sn = _mm_cmplt_epi16(sn, zero);

  /* if ( sn) { snAluOp = (vs + vt); } */
  /* if (!sn) { snAluOp = (vs - vt); } */
  snAluOp = _mm_xor_si128(vtShuf, sn);
  negVtReg = _mm_add_epi16(snAluOp, sn);
  snAluOp = _mm_sub_epi16(vsReg, negVtReg);

  /* Compute ge, le for each case. */
  /* if ( sn) { ge = (vt < 0);       le = (vs + vt <= 0); */
  /* if (!sn) { ge = (vs - vt >= 0); le = (vt < 0);       */
  cmp1 = _mm_cmplt_epi16(vtShuf, zero);
  cmp2 = _mm_cmplt_epi16(snAluOp, zero);
  cmp2 = _mm_cmpeq_epi16(cmp2, sn);
  temp = _mm_cmpeq_epi16(snAluOp, zero);
  cmp2 = _mm_or_si128(cmp2, temp);

#ifdef SSSE3_ONLY
  temp2a = _mm_and_si128(sn, cmp1);
  temp2s = _mm_andnot_si128(sn, cmp2);
  ge = _mm_or_si128(temp2a, temp2s);
  temp2a = _mm_and_si128(sn, cmp2);
  temp2s = _mm_andnot_si128(sn, cmp1);
  le = _mm_or_si128(temp2a, temp2s);
#else
  ge = _mm_blendv_epi8(cmp2, cmp1, sn);
  le = _mm_blendv_epi8(cmp1, cmp2, sn);
#endif

  /* Compute neq, vce for each case. */
  /* if ( sn) { neq = (vs + vt == -1); vce |= neq;  neq ^= !(vs + vt == 0); } */
  /* if (!sn) { neq = !(vs - vt == 0); vce |= 0x00;                         } */
  temp = _mm_cmpeq_epi16(snAluOp, sn);
  temp = _mm_and_si128(temp, sn);
  _mm_store_si128((__m128i*) cp2->vce.slices, temp);

  temp = _mm_cmpeq_epi16(snAluOp, zero);
  neq = _mm_cmpeq_epi16(temp, zero);

  /* Compute accLow for each case. */
  /* if ( sn) { accLow = le ? -vt : vs; */
  /* if (!sn) { accLow = ge ?  vt : vs; */
#ifdef SSSE3_ONLY
  temp2a = _mm_and_si128(cmp2, negVtReg);
  temp2s = _mm_andnot_si128(cmp2, vsReg);
  temp = _mm_or_si128(temp2a, temp2s);
#else
  temp = _mm_blendv_epi8(vsReg, negVtReg, cmp2);
#endif

  /* Compute vco, vcc for each case. */
  /* vcchi |=  ge; vcolo |= le;  */
  /* vcohi |= neq; vcolo |= sn;  */
  _mm_store_si128((__m128i*) (cp2->vcclo.slices), le);
  _mm_store_si128((__m128i*) (cp2->vcchi.slices), ge);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), sn);
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), neq);
  _mm_store_si128((__m128i*) accLow, temp);
  return temp;
#else
#warning "Unimplemented function: RSPVCH (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VCL (Vector Select Clip Test Low)
 * ========================================================================= */
__m128i
RSPVCL(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i ce, eq, ge, le, lz, mask, negVtReg, sn;
  __m128i result, satsum, sum, temp1, temp2, uz;
  __m128i newge, newle;

  ce = _mm_load_si128((__m128i*) (cp2->vce.slices));
  eq = _mm_load_si128((__m128i*) (cp2->vcohi.slices));
  ge = _mm_load_si128((__m128i*) (cp2->vcchi.slices));
  le = _mm_load_si128((__m128i*) (cp2->vcclo.slices));
  sn = _mm_load_si128((__m128i*) (cp2->vcolo.slices));
  eq = _mm_cmpeq_epi16(eq, zero); /* ~eq */

  /* (sn) ? -vt : vt */
  negVtReg = _mm_xor_si128(vtShuf, sn);
  negVtReg = _mm_sub_epi16(negVtReg, sn);

  /* lz = ((sum & 0x0000FFFF) == 0x00000000 */
  /* uz = ((sum & 0xFFFF0000) == 0x00000000 */
  satsum = _mm_adds_epu16(vsReg, vtShuf);
  sum = _mm_add_epi16(vsReg, vtShuf);
  uz = _mm_cmplt_epi16(sum, satsum);
  uz = _mm_cmpeq_epi16(uz, zero);
  lz = _mm_cmpeq_epi16(sum, zero);

  /* if (sn && eq) le = ((!ce) & (lz & uz)) | (ce & (lz | uz)) */
  mask = _mm_and_si128(sn, eq);
  temp1 = _mm_or_si128(lz, uz);
  result = _mm_and_si128(ce, temp1);
  temp1 = _mm_cmpeq_epi16(ce, zero);
  temp2 = _mm_and_si128(lz, uz);
  temp1 = _mm_and_si128(temp1, temp2);
  temp2 = _mm_or_si128(result, temp1);
#ifdef SSSE3_ONLY
  newle = _mm_and_si128(mask, temp2);
  le = _mm_andnot_si128(mask, le);
  le = _mm_or_si128(le, newle);
#else
  newle = _mm_blendv_epi8(le, temp2, mask);
#endif

  /* if (!sn && eq) ge = ((vs - vt) >= 0) */
  mask = _mm_andnot_si128(sn, eq);
  temp2 = _mm_subs_epu16(vsReg, vtShuf);
  temp1 = _mm_adds_epu16(temp2, vtShuf);
  temp1 = _mm_cmpeq_epi16(temp1, vsReg);
#ifdef SSSE3_ONLY
  newge = _mm_and_si128(mask, temp1);
  ge = _mm_andnot_si128(mask, ge);
  ge = _mm_or_si128(ge, newge);
#else
  newge = _mm_blendv_epi8(ge, temp1, mask);
#endif

  /* selector = sn ? le : ge; */
  /* acc = selector ? {-}vt : vs; */
#ifdef SSSE3_ONLY
  temp1 = _mm_and_si128(sn, le);
  temp2 = _mm_andnot_si128(sn, ge);
  mask = _mm_or_si128(temp1, temp2);
#else
  mask = _mm_blendv_epi8(ge, le, sn);
#endif

#ifdef SSSE3_ONLY
  temp1 = _mm_and_si128(mask, negVtReg);
  temp2 = _mm_andnot_si128(mask, vsReg);
  result = _mm_or_si128(temp1, temp2);
#else
  result = _mm_blendv_epi8(vsReg, negVtReg, mask);
#endif

  _mm_storeu_si128((__m128i*) accLow, result);
  _mm_store_si128((__m128i*) (cp2->vcclo.slices), le);
  _mm_store_si128((__m128i*) (cp2->vcchi.slices), ge);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), zero);
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), zero);
  _mm_store_si128((__m128i*) (cp2->vce.slices), zero);
  return result;
#else
#warning "Unimplemented function: RSPVCL (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VCR (Vector Select Crimp Test Low)
 * ========================================================================= */
__m128i
RSPVCR(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i cmp1, cmp2, negVtReg, notVtReg, snAluOp, temp, temp2a, temp2s;
  __m128i ge, le, sn;

  /* sn = (vs ^ vt) < 0 */
  sn = _mm_xor_si128(vsReg, vtShuf);
  sn = _mm_cmplt_epi16(sn, zero);

  /* if ( sn) { snAluOp = (vs + vt); } */
  /* if (!sn) { snAluOp = (vs - vt); } */
  notVtReg = _mm_xor_si128(vtShuf, sn);
  negVtReg = _mm_sub_epi16(notVtReg, sn);
  snAluOp = _mm_sub_epi16(vsReg, negVtReg);

  /* Compute ge, le for each case. */
  /* if ( sn) { ge = (vt < 0);       le = (vs + vt < 0); */
  /* if (!sn) { ge = (vs - vt >= 0); le = (vt < 0);      */
  cmp1 = _mm_cmplt_epi16(vtShuf, zero);
  cmp2 = _mm_cmplt_epi16(snAluOp, zero);
  cmp2 = _mm_cmpeq_epi16(cmp2, sn);

#ifdef SSSE3_ONLY
  temp2a = _mm_and_si128(sn, cmp1);
  temp2s = _mm_andnot_si128(sn, cmp2);
  ge = _mm_or_si128(temp2a, temp2s);
  temp2a = _mm_and_si128(sn, cmp2);
  temp2s = _mm_andnot_si128(sn, cmp1);
  le = _mm_or_si128(temp2a, temp2s);
#else
  ge = _mm_blendv_epi8(cmp2, cmp1, sn);
  le = _mm_blendv_epi8(cmp1, cmp2, sn);
#endif

  /* Compute accLow for each case. */
  /* if ( sn) { accLow = le ? ~vt : vs; */
  /* if (!sn) { accLow = le ?  vt : vs; */
#ifdef SSSE3_ONLY
  temp2a = _mm_and_si128(le, notVtReg);
  temp2s = _mm_andnot_si128(le, vsReg);
  temp = _mm_or_si128(temp2a, temp2s);
#else
  temp = _mm_blendv_epi8(vsReg, negVtReg, le);
#endif

  _mm_store_si128((__m128i*) accLow, temp);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), zero);
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), zero);
  _mm_store_si128((__m128i*) (cp2->vcclo.slices), le);
  _mm_store_si128((__m128i*) (cp2->vcchi.slices), ge);
  _mm_store_si128((__m128i*) (cp2->vce.slices), zero);
  return temp;
#else
#warning "Unimplemented function: RSPVCR (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VEQ (Vector Select Equal)
 * ========================================================================= */
__m128i
RSPVEQ(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vne = _mm_load_si128((__m128i*) (cp2->vcohi.slices));
  __m128i equal = _mm_cmpeq_epi16(vtShuf, vsReg);
  vne = _mm_cmpeq_epi16(vne, _mm_setzero_si128());
  __m128i vvcc = _mm_and_si128(equal, vne);

  _mm_store_si128((__m128i*) accLow, vtShuf);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), zero);
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), zero);
  _mm_store_si128((__m128i*) (cp2->vcclo.slices), vvcc);
  _mm_store_si128((__m128i*) (cp2->vcchi.slices), zero);
  return vtShuf;
#else
#warning "Unimplemented function: RSPVEQ (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VGE (Vector Select Greater Than or Equal)
 * ========================================================================= */
__m128i
RSPVGE(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vne = _mm_load_si128((__m128i*) (cp2->vcohi.slices));
  __m128i vco = _mm_load_si128((__m128i*) (cp2->vcolo.slices));
  __m128i temp, equal, greaterEqual, vdReg;

  /* equal = (~vco | ~vne) && (vs == vt) */
  temp = _mm_and_si128(vne, vco);
  temp = _mm_cmpeq_epi16(temp, zero);
  equal = _mm_cmpeq_epi16(vsReg, vtShuf);
  equal = _mm_and_si128(temp, equal);

  /* ge = vs > vt | equal */
  greaterEqual = _mm_cmpgt_epi16(vsReg, vtShuf);
  greaterEqual = _mm_or_si128(greaterEqual, equal);

  /* vd = ge ? vs : vt; */
#ifdef SSSE3_ONLY
  vsReg = _mm_and_si128(greaterEqual, vsReg);
  vtShuf = _mm_andnot_si128(greaterEqual, vtShuf);
  vdReg = _mm_or_si128(vsReg, vtShuf);
#else
  vdReg = _mm_blendv_epi8(vtShuf, vsReg, greaterEqual);
#endif

  _mm_store_si128((__m128i*) accLow, vdReg);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), zero);
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), zero);
  _mm_store_si128((__m128i*) (cp2->vcclo.slices), greaterEqual);
  _mm_store_si128((__m128i*) (cp2->vcchi.slices), zero);
  return vdReg;
#else
#warning "Unimplemented function: RSPVGE (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VINV (Invalid Vector Operation)
 * ========================================================================= */
__m128i
RSPVINV(struct RSPCP2 *unused(cp2), int16_t *vd, __m128i unused(vsReg),
  __m128i unused(vtReg), __m128i unused(vtShuf), __m128i unused(zero)) {
  return _mm_load_si128((__m128i*) vd);
}

/* ============================================================================
 *  Instruction: VLT (Vector Select Less Than or Equal)
 * ========================================================================= */
__m128i
RSPVLT(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vne = _mm_load_si128((__m128i*) (cp2->vcohi.slices));
  __m128i vco = _mm_load_si128((__m128i*) (cp2->vcolo.slices));
  __m128i temp, equal, lessthanEqual, vdReg;

  /* equal = (vco & vne) && (vs == vt) */
  temp = _mm_and_si128(vne, vco);
  equal = _mm_cmpeq_epi16(vsReg, vtShuf);
  equal = _mm_and_si128(equal, temp);

  /* le = vs < vt | equal */
  lessthanEqual = _mm_cmplt_epi16(vsReg, vtShuf);
  lessthanEqual = _mm_or_si128(lessthanEqual, equal);

  /* vd = le ? vs : vt; */
#ifdef SSSE3_ONLY
  vsReg = _mm_and_si128(lessthanEqual, vsReg);
  vtShuf = _mm_andnot_si128(lessthanEqual, vtShuf);
  vdReg = _mm_or_si128(vsReg, vtShuf);
#else
  vdReg = _mm_blendv_epi8(vtShuf, vsReg, lessthanEqual);
#endif

  _mm_store_si128((__m128i*) accLow, vdReg);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), zero);
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), zero);
  _mm_store_si128((__m128i*) (cp2->vcclo.slices), lessthanEqual);
  _mm_store_si128((__m128i*) (cp2->vcchi.slices), zero);
  return vdReg;
#else
#warning "Unimplemented function: RSPVLT (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VMACF (Vector Multiply-Accumulate of Signed Fractions)
 * ========================================================================= */
__m128i
RSPVMACF(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;

#ifdef USE_SSE
  __m128i loProduct, hiProduct, unpackLo, unpackHi;
  __m128i vaccTemp, vaccLow, vaccMid, vaccHigh;
  __m128i vdReg, vdRegLo, vdRegHi;

  vaccLow = _mm_load_si128((__m128i*) accLow);
  vaccMid = _mm_load_si128((__m128i*) accMid);

  /* Unpack to obtain for 32-bit precision. */
  RSPZeroExtend16to32(vaccLow, &vaccLow, &vaccHigh, zero);

  /* Begin accumulating the products. */
  unpackLo = _mm_mullo_epi16(vsReg, vtShuf);
  unpackHi = _mm_mulhi_epi16(vsReg, vtShuf);
  loProduct = _mm_unpacklo_epi16(unpackLo, unpackHi);
  hiProduct = _mm_unpackhi_epi16(unpackLo, unpackHi);
  loProduct = _mm_slli_epi32(loProduct, 1);
  hiProduct = _mm_slli_epi32(hiProduct, 1);

#ifdef SSSE3_ONLY
  vdRegLo = _mm_srli_epi32(loProduct, 16);
  vdRegHi = _mm_srli_epi32(hiProduct, 16);
  vdRegLo = _mm_slli_epi32(vdRegLo, 16);
  vdRegHi = _mm_slli_epi32(vdRegHi, 16);
  vdRegLo = _mm_xor_si128(vdRegLo, loProduct);
  vdRegHi = _mm_xor_si128(vdRegHi, hiProduct);
#else
  vdRegLo = _mm_blend_epi16(loProduct, zero, 0xAA);
  vdRegHi = _mm_blend_epi16(hiProduct, zero, 0xAA);
#endif

  vaccLow = _mm_add_epi32(vaccLow, vdRegLo);
  vaccHigh = _mm_add_epi32(vaccHigh, vdRegHi);

  vdReg = RSPPackLo32to16(vaccLow, vaccHigh, zero);
  _mm_store_si128((__m128i*) accLow, vdReg);

  /* Multiply the MSB of sources, accumulate the product. */
  vaccTemp = _mm_load_si128((__m128i*) accHigh);
  vdRegLo = _mm_unpacklo_epi16(vaccMid, vaccTemp);
  vdRegHi = _mm_unpackhi_epi16(vaccMid, vaccTemp);

  loProduct = _mm_srai_epi32(loProduct, 16);
  hiProduct = _mm_srai_epi32(hiProduct, 16);
  vaccLow = _mm_srai_epi32(vaccLow, 16);
  vaccHigh = _mm_srai_epi32(vaccHigh, 16);

  vaccLow = _mm_add_epi32(loProduct, vaccLow);
  vaccHigh = _mm_add_epi32(hiProduct, vaccHigh);
  vaccLow = _mm_add_epi32(vdRegLo, vaccLow);
  vaccHigh = _mm_add_epi32(vdRegHi, vaccHigh);

  /* Clamp the accumulator and write it all out. */
  vdReg = _mm_packs_epi32(vaccLow, vaccHigh);
  RSPPack32to16(vaccLow, vaccHigh, &vaccMid, &vaccHigh);

  _mm_store_si128((__m128i*) accMid, vaccMid);
  _mm_store_si128((__m128i*) accHigh, vaccHigh);
  return vdReg;
#else
#warning "Unimplemented function: VMACF (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VMACQ (Vector Accumulator Oddification)
 * ========================================================================= */
__m128i
RSPVMACQ(struct RSPCP2 *unused(cp2), int16_t *unused(vd), __m128i unused(vsReg),
  __m128i unused(vtReg), __m128i unused(vtShuf), __m128i zero) {
  debug("Unimplemented function: VMACQ.");
  return zero;
}

/* ============================================================================
 *  Instruction: VMACU (Vector Multiply-Accumulate of Unsigned Fractions)
 * ========================================================================= */
__m128i
RSPVMACU(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;
  unsigned i;

#ifdef USE_SSE
  __m128i loProduct, hiProduct, unpackLo, unpackHi;;
  __m128i vaccTemp, vaccLow, vaccMid, vaccHigh;
  __m128i vdReg, vdRegLo, vdRegHi;
  vaccLow = _mm_load_si128((__m128i*) accLow);
  vaccMid = _mm_load_si128((__m128i*) accMid);

  /* Unpack to obtain for 32-bit precision. */
  RSPZeroExtend16to32(vaccLow, &vaccLow, &vaccHigh, zero);

  /* Begin accumulating the products. */
  unpackLo = _mm_mullo_epi16(vsReg, vtShuf);
  unpackHi = _mm_mulhi_epi16(vsReg, vtShuf);
  loProduct = _mm_unpacklo_epi16(unpackLo, unpackHi);
  hiProduct = _mm_unpackhi_epi16(unpackLo, unpackHi);
  loProduct = _mm_slli_epi32(loProduct, 1);
  hiProduct = _mm_slli_epi32(hiProduct, 1);

#ifdef SSSE3_ONLY
  vdRegLo = _mm_srli_epi32(loProduct, 16);
  vdRegHi = _mm_srli_epi32(hiProduct, 16);
  vdRegLo = _mm_slli_epi32(vdRegLo, 16);
  vdRegHi = _mm_slli_epi32(vdRegHi, 16);
  vdRegLo = _mm_xor_si128(vdRegLo, loProduct);
  vdRegHi = _mm_xor_si128(vdRegHi, hiProduct);
#else
  vdRegLo = _mm_blend_epi16(loProduct, zero, 0xAA);
  vdRegHi = _mm_blend_epi16(hiProduct, zero, 0xAA);
#endif

  vaccLow = _mm_add_epi32(vaccLow, vdRegLo);
  vaccHigh = _mm_add_epi32(vaccHigh, vdRegHi);

  vdReg = RSPPackLo32to16(vaccLow, vaccHigh, zero);
  _mm_store_si128((__m128i*) accLow, vdReg);

  /* Multiply the MSB of sources, accumulate the product. */
  vaccTemp = _mm_load_si128((__m128i*) accHigh);
  vdRegLo = _mm_unpacklo_epi16(vaccMid, vaccTemp);
  vdRegHi = _mm_unpackhi_epi16(vaccMid, vaccTemp);

  loProduct = _mm_srai_epi32(loProduct, 16);
  hiProduct = _mm_srai_epi32(hiProduct, 16);
  vaccLow = _mm_srai_epi32(vaccLow, 16);
  vaccHigh = _mm_srai_epi32(vaccHigh, 16);

  vaccLow = _mm_add_epi32(loProduct, vaccLow);
  vaccHigh = _mm_add_epi32(hiProduct, vaccHigh);
  vaccLow = _mm_add_epi32(vdRegLo, vaccLow);
  vaccHigh = _mm_add_epi32(vdRegHi, vaccHigh);

  /* Clamp the accumulator and write it all out. */
  RSPPack32to16(vaccLow, vaccHigh, &vaccMid, &vaccHigh);
  _mm_store_si128((__m128i*) accMid, vaccMid);
  _mm_store_si128((__m128i*) accHigh, vaccHigh);
#else
#warning "Unimplemented function: VMACU (No SSE)."
#endif

  for (i = 0; i < 8; i++) {
    signed short result;
    short int tmp;

    result = accMid[i];

    unsigned long long lopiece = (unsigned short) accLow[i];
    unsigned long long mdpiece = (unsigned short) accMid[i];
    unsigned long long hipiece = (unsigned short) accHigh[i];
    mdpiece <<= 16;
    hipiece <<= 32;

    signed long long int thing = lopiece | mdpiece | hipiece;
    tmp = (signed short)(thing >> 31) != 0x0000;
    result |= -tmp;
    tmp = accHigh[i] >> 15;
    result &= ~tmp;
    vd[i] = result;
  }

  return _mm_load_si128((__m128i*) vd);
}

/* ============================================================================
 *  Instruction: VMADH (Vector Multiply-Accumulate of High Partial Products)
 * ========================================================================= */
__m128i
RSPVMADH(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i unused(zero)) {
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;

#ifdef USE_SSE
  __m128i unpackLo, unpackHi, loProduct, hiProduct;
  __m128i vaccLow, vaccMid, vaccHigh, vdReg;
  vaccMid = _mm_load_si128((__m128i*) accMid);
  vaccHigh = _mm_load_si128((__m128i*) accHigh);

  /* Unpack to obtain for 32-bit precision. */
  vaccLow = _mm_unpacklo_epi16(vaccMid, vaccHigh);
  vaccHigh = _mm_unpackhi_epi16(vaccMid, vaccHigh);

  /* Multiply the sources, accumulate the product. */
  unpackLo = _mm_mullo_epi16(vsReg, vtShuf);
  unpackHi = _mm_mulhi_epi16(vsReg, vtShuf);
  loProduct = _mm_unpacklo_epi16(unpackLo, unpackHi);
  hiProduct = _mm_unpackhi_epi16(unpackLo, unpackHi);
  vaccLow = _mm_add_epi32(vaccLow, loProduct);
  vaccHigh = _mm_add_epi32(vaccHigh, hiProduct);

  /* Pack the accumulator and result back up. */
  vdReg = _mm_packs_epi32(vaccLow, vaccHigh);
  RSPPack32to16(vaccLow, vaccHigh, &vaccMid, &vaccHigh);

  _mm_store_si128((__m128i*) accMid, vaccMid);
  _mm_store_si128((__m128i*) accHigh, vaccHigh);
  return vdReg;
#else
#warning "Unimplemented function: RSPVMADH (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VMADL (Vector Multiply-Accumulate of Lower Partial Products).
 * ========================================================================= */
__m128i
RSPVMADL(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;

#ifdef USE_SSE
  __m128i vaccTemp, vaccLow, vaccMid, vaccHigh;
  __m128i unpackHi, loProduct, hiProduct;
  __m128i vdReg, vdRegLo, vdRegHi;
  vaccLow = _mm_load_si128((__m128i*) accLow);
  vaccMid = _mm_load_si128((__m128i*) accMid);

  /* Unpack to obtain for 32-bit precision. */
  RSPZeroExtend16to32(vaccLow, &vaccLow, &vaccHigh, zero);

  /* Begin accumulating the products. */
  unpackHi = _mm_mulhi_epu16(vsReg, vtShuf);
  loProduct = _mm_unpacklo_epi16(unpackHi, zero);
  hiProduct = _mm_unpackhi_epi16(unpackHi, zero);

  vaccLow = _mm_add_epi32(vaccLow, loProduct);
  vaccHigh = _mm_add_epi32(vaccHigh, hiProduct);
  vdReg = RSPPackLo32to16(vaccLow, vaccHigh, zero);
  _mm_store_si128((__m128i*) accLow, vdReg);

  /* Finish accumulating whatever is left. */
  vaccTemp = _mm_load_si128((__m128i*) accHigh);
  vdRegLo = _mm_unpacklo_epi16(vaccMid, vaccTemp);
  vdRegHi = _mm_unpackhi_epi16(vaccMid, vaccTemp);

  vaccLow = _mm_srai_epi32(vaccLow, 16);
  vaccHigh = _mm_srai_epi32(vaccHigh, 16);
  vaccLow = _mm_add_epi32(vdRegLo, vaccLow);
  vaccHigh = _mm_add_epi32(vdRegHi, vaccHigh);

  /* Clamp the accumulator and write it all out. */
  RSPPack32to16(vaccLow, vaccHigh, &vaccMid, &vaccHigh);
  vdReg = RSPClampLowToVal(vdReg, vaccMid, vaccHigh, zero);

  _mm_store_si128((__m128i*) accMid, vaccMid);
  _mm_store_si128((__m128i*) accHigh, vaccHigh);
  return vdReg;
#else
#warning "Unimplemented function: RSPVMADL (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VMADM (Vector Multiply-Accumulate of Mid Partial Products)
 * ========================================================================= */
__m128i
RSPVMADM(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;

#ifdef USE_SSE
  __m128i vaccTemp, vaccLow, vaccMid, vaccHigh, loProduct, hiProduct;
  __m128i vsRegLo, vsRegHi, vtRegLo, vtRegHi, vdReg, vdRegLo, vdRegHi;
  vaccLow = _mm_load_si128((__m128i*) accLow);
  vaccMid = _mm_load_si128((__m128i*) accMid);

  /* Unpack to obtain for 32-bit precision. */
  RSPSignExtend16to32(vsReg, &vsRegLo, &vsRegHi);
  RSPZeroExtend16to32(vtShuf, &vtRegLo, &vtRegHi, zero);
  RSPZeroExtend16to32(vaccLow, &vaccLow, &vaccHigh, zero);

  /* Begin accumulating the products. */
  loProduct = _mm_mullo_epi32(vsRegLo, vtRegLo);
  hiProduct = _mm_mullo_epi32(vsRegHi, vtRegHi);

#ifdef SSSE3_ONLY
  vdRegLo = _mm_srli_epi32(loProduct, 16);
  vdRegHi = _mm_srli_epi32(hiProduct, 16);
  vdRegLo = _mm_slli_epi32(vdRegLo, 16);
  vdRegHi = _mm_slli_epi32(vdRegHi, 16);
  vdRegLo = _mm_xor_si128(vdRegLo, loProduct);
  vdRegHi = _mm_xor_si128(vdRegHi, hiProduct);
#else
  vdRegLo = _mm_blend_epi16(loProduct, zero, 0xAA);
  vdRegHi = _mm_blend_epi16(hiProduct, zero, 0xAA);
#endif
  vaccLow = _mm_add_epi32(vaccLow, vdRegLo);
  vaccHigh = _mm_add_epi32(vaccHigh, vdRegHi);

  vdReg = RSPPackLo32to16(vaccLow, vaccHigh, zero);
  _mm_store_si128((__m128i*) accLow, vdReg);

  /* Multiply the MSB of sources, accumulate the product. */
  vaccTemp = _mm_load_si128((__m128i*) accHigh);
  vdRegLo = _mm_unpacklo_epi16(vaccMid, vaccTemp);
  vdRegHi = _mm_unpackhi_epi16(vaccMid, vaccTemp);

  loProduct = _mm_srai_epi32(loProduct, 16);
  hiProduct = _mm_srai_epi32(hiProduct, 16);
  vaccLow = _mm_srai_epi32(vaccLow, 16);
  vaccHigh = _mm_srai_epi32(vaccHigh, 16);

  vaccLow = _mm_add_epi32(loProduct, vaccLow);
  vaccHigh = _mm_add_epi32(hiProduct, vaccHigh);
  vaccLow = _mm_add_epi32(vdRegLo, vaccLow);
  vaccHigh = _mm_add_epi32(vdRegHi, vaccHigh);

  /* Clamp the accumulator and write it all out. */
  vdReg = _mm_packs_epi32(vaccLow, vaccHigh);
  RSPPack32to16(vaccLow, vaccHigh, &vaccMid, &vaccHigh);

  _mm_store_si128((__m128i*) accMid, vaccMid);
  _mm_store_si128((__m128i*) accHigh, vaccHigh);
  return vdReg;
#else
#warning "Unimplemented function: RSPVMADM (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VMADN (Vector Multiply-Accumulate of Mid Partial Products)
 * ========================================================================= */
__m128i
RSPVMADN(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;

#ifdef USE_SSE
  __m128i vaccTemp, vaccLow, vaccMid, vaccHigh, loProduct, hiProduct;
  __m128i vsRegLo, vsRegHi, vtRegLo, vtRegHi, vdReg, vdRegLo, vdRegHi;
  vaccLow = _mm_load_si128((__m128i*) accLow);
  vaccMid = _mm_load_si128((__m128i*) accMid);

  /* Unpack to obtain for 32-bit precision. */
  RSPSignExtend16to32(vtShuf, &vtRegLo, &vtRegHi);
  RSPZeroExtend16to32(vsReg, &vsRegLo, &vsRegHi, zero);
  RSPZeroExtend16to32(vaccLow, &vaccLow, &vaccHigh, zero);

  /* Begin accumulating the products. */
  loProduct = _mm_mullo_epi32(vsRegLo, vtRegLo);
  hiProduct = _mm_mullo_epi32(vsRegHi, vtRegHi);

#ifdef SSSE3_ONLY
  vdRegLo = _mm_srli_epi32(loProduct, 16);
  vdRegHi = _mm_srli_epi32(hiProduct, 16);
  vdRegLo = _mm_slli_epi32(vdRegLo, 16);
  vdRegHi = _mm_slli_epi32(vdRegHi, 16);
  vdRegLo = _mm_xor_si128(vdRegLo, loProduct);
  vdRegHi = _mm_xor_si128(vdRegHi, hiProduct);
#else
  vdRegLo = _mm_blend_epi16(loProduct, zero, 0xAA);
  vdRegHi = _mm_blend_epi16(hiProduct, zero, 0xAA);
#endif

  vaccLow = _mm_add_epi32(vaccLow, vdRegLo);
  vaccHigh = _mm_add_epi32(vaccHigh, vdRegHi);

   vdReg = RSPPackLo32to16(vaccLow, vaccHigh, zero);
  _mm_store_si128((__m128i*) accLow, vdReg);

  /* Multiply the MSB of sources, accumulate the product. */
  vaccTemp = _mm_load_si128((__m128i*) accHigh);
  vdRegLo = _mm_unpacklo_epi16(vaccMid, vaccTemp);
  vdRegHi = _mm_unpackhi_epi16(vaccMid, vaccTemp);

  loProduct = _mm_srai_epi32(loProduct, 16);
  hiProduct = _mm_srai_epi32(hiProduct, 16);
  vaccLow = _mm_srai_epi32(vaccLow, 16);
  vaccHigh = _mm_srai_epi32(vaccHigh, 16);

  vaccLow = _mm_add_epi32(loProduct, vaccLow);
  vaccHigh = _mm_add_epi32(hiProduct, vaccHigh);
  vaccLow = _mm_add_epi32(vdRegLo, vaccLow);
  vaccHigh = _mm_add_epi32(vdRegHi, vaccHigh);

  /* Clamp the accumulator and write it all out. */
  RSPPack32to16(vaccLow, vaccHigh, &vaccMid, &vaccHigh);
  vdReg = RSPClampLowToVal(vdReg, vaccMid, vaccHigh, zero);

  _mm_store_si128((__m128i*) accMid, vaccMid);
  _mm_store_si128((__m128i*) accHigh, vaccHigh);
  return vdReg;
#else
#warning "Unimplemented function: RSPVMADN (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VMOV (Vector Element Scalar Move)
 * ========================================================================= */
__m128i
RSPVMOV(struct RSPCP2 *cp2, int16_t *vd, __m128i unused(vsReg),
  __m128i unused(vtReg), __m128i vtShuf, __m128i unused(zero)) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  unsigned delement = cp2->iw >> 11 & 0x1F;

#ifdef USE_SSE
  _mm_store_si128((__m128i*) accLow, vtShuf);
#else
#warning "Unimplemented function: RSPVMOV (No SSE)."
#endif

  vd[delement & 0x7] = accLow[delement & 0x7];
  return _mm_load_si128((__m128i*) vd);
}

/* ============================================================================
 *  Instruction: VMRG (Vector Select Merge)
 * ========================================================================= */
__m128i
RSPVMRG(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i unused(zero)) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i mask = _mm_load_si128((__m128i*) (cp2->vcclo.slices));

#ifdef SSSE3_ONLY
  __m128i temp1 = _mm_and_si128(mask, vsReg);
  __m128i temp2 = _mm_andnot_si128(mask, vtShuf);
  __m128i vaccLow = _mm_or_si128(temp1, temp2);
#else
  __m128i vaccLow = _mm_blendv_epi8(vtShuf, vsReg, mask);
#endif

  _mm_store_si128((__m128i*) accLow, vaccLow);
  return vaccLow;
#else
#warning "Unimplemented function: RSPVMRG (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VMUDH (Vector Multiply of High Partial Products)
 * ========================================================================= */
__m128i
RSPVMUDH(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;

#ifdef USE_SSE
  __m128i vaccLow, vaccMid, vaccHigh, vdReg;
  __m128i unpackLo, unpackHi;

  /* Multiply the sources, accumulate the product. */
  unpackLo = _mm_mullo_epi16(vsReg, vtShuf);
  unpackHi = _mm_mulhi_epi16(vsReg, vtShuf);
  vaccHigh = _mm_unpackhi_epi16(unpackLo, unpackHi);
  vaccLow = _mm_unpacklo_epi16(unpackLo, unpackHi);

  /* Pack the accumulator and result back up. */
  vdReg = _mm_packs_epi32(vaccLow, vaccHigh);
  RSPPack32to16(vaccLow, vaccHigh, &vaccMid, &vaccHigh);

  _mm_store_si128((__m128i*) accLow, zero);
  _mm_store_si128((__m128i*) accMid, vaccMid);
  _mm_store_si128((__m128i*) accHigh, vaccHigh);
  return vdReg;
#else
  debug("Unimplemented function: VMUDH.");
#endif
}

/* ============================================================================
 *  Instruction: VMUDL (Vector Multiply of Low Partial Products)
 * ========================================================================= */
__m128i
RSPVMUDL(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vt), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;

#ifdef USE_SSE
  __m128i unpackLo, unpackHi, loProduct, hiProduct, vdReg;

  /* Unpack to obtain for 32-bit precision. */
  unpackLo = _mm_mullo_epi16(vsReg, vtShuf);
  unpackHi = _mm_mulhi_epu16(vsReg, vtShuf);
  loProduct = _mm_unpacklo_epi16(unpackLo, unpackHi);
  hiProduct = _mm_unpackhi_epi16(unpackLo, unpackHi);
  vdReg = RSPPackHi32to16(loProduct, hiProduct);

  _mm_store_si128((__m128i*) accLow, vdReg);
  _mm_store_si128((__m128i*) accMid, zero);
  _mm_store_si128((__m128i*) accHigh, zero);
#else
  debug("Unimplemented function: VMUDL.");
#endif
  return vdReg;
}

/* ============================================================================
 *  Instruction: VMUDM (Vector Multiply of Middle Partial Products)
 * ========================================================================= */
__m128i
RSPVMUDM(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;

#ifdef USE_SSE
  __m128i vsRegLo, vsRegHi, vtRegLo, vtRegHi, vdReg;
  __m128i loProduct, hiProduct, vaccLow, vaccHigh;

  /* Unpack to obtain for 32-bit precision. */
  RSPSignExtend16to32(vsReg, &vsRegLo, &vsRegHi);
  RSPZeroExtend16to32(vtShuf, &vtRegLo, &vtRegHi, zero);

  /* Begin accumulating the products. */
  loProduct = _mm_mullo_epi32(vsRegLo, vtRegLo);
  hiProduct = _mm_mullo_epi32(vsRegHi, vtRegHi);
  RSPPack32to16(loProduct, hiProduct, &vaccLow, &vaccHigh);

  loProduct = _mm_cmplt_epi32(loProduct, zero);
  hiProduct = _mm_cmplt_epi32(hiProduct, zero);
  vdReg = _mm_packs_epi32(loProduct, hiProduct);

  _mm_store_si128((__m128i*) accLow, vaccLow);
  _mm_store_si128((__m128i*) accMid, vaccHigh);
  _mm_store_si128((__m128i*) accHigh, vdReg);
  return vaccHigh;
#else
  debug("Unimplemented function: VMUDM.");
#endif
}

/* ============================================================================
 *  Instruction: VMUDN (Vector Multiply of Middle Partial Products)
 * ========================================================================= */
__m128i
RSPVMUDN(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;

#ifdef USE_SSE
  __m128i vsRegLo, vsRegHi, vtRegLo, vtRegHi, vdReg;
  __m128i loProduct, hiProduct, vaccLow, vaccHigh;

  /* Unpack to obtain for 32-bit precision. */
  RSPZeroExtend16to32(vsReg, &vsRegLo, &vsRegHi, zero);
  RSPSignExtend16to32(vtShuf, &vtRegLo, &vtRegHi);

  /* Begin accumulating the products. */
  loProduct = _mm_mullo_epi32(vsRegLo, vtRegLo);
  hiProduct = _mm_mullo_epi32(vsRegHi, vtRegHi);
  RSPPack32to16(loProduct, hiProduct, &vaccLow, &vaccHigh);
  vdReg = _mm_cmplt_epi16(vaccHigh, zero);

  _mm_store_si128((__m128i*) accLow, vaccLow);
  _mm_store_si128((__m128i*) accMid, vaccHigh);
  _mm_store_si128((__m128i*) accHigh, vdReg);
  return vaccLow;
#else
  debug("Unimplemented function: VMUDN.");
#endif
}

/* ============================================================================
 *  Instruction: VMULF (Vector Multiply of Signed Fractions).
 * ========================================================================= */
__m128i
RSPVMULF(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;

#ifdef USE_SSE
  __m128i lowProduct, highProduct, unpackLow, unpackHigh;
  __m128i temp, vaccLow, vaccMid, vaccHigh, vdReg;

  /* Compute 64-bit signed product. */
  unpackLow = _mm_mullo_epi16(vsReg, vtShuf);
  unpackHigh = _mm_mulhi_epi16(vsReg, vtShuf);
  lowProduct = _mm_unpacklo_epi16(unpackLow, unpackHigh);
  highProduct = _mm_unpackhi_epi16(unpackLow, unpackHigh);
  lowProduct = _mm_slli_epi32(lowProduct, 1);
  highProduct = _mm_slli_epi32(highProduct, 1);

  /* Add the rounding value. */
  temp = _mm_set1_epi32(0x00008000U);
  lowProduct = _mm_add_epi32(lowProduct, temp);
  highProduct = _mm_add_epi32(highProduct, temp);

  /* Compute the accmulator, paying attention to overflow. */
  RSPPack32to16(lowProduct, highProduct, &vaccLow, &vaccMid);
  vaccHigh = _mm_cmplt_epi16(vaccMid, zero);
  temp = _mm_cmpeq_epi16(vsReg, vtShuf);
  temp = _mm_cmpeq_epi16(temp, zero);
  vaccHigh = _mm_and_si128(vaccHigh, temp);

  /* Compute the result by clamping the middle word. */
  lowProduct = _mm_unpacklo_epi16(vaccMid, vaccHigh);
  highProduct = _mm_unpackhi_epi16(vaccMid, vaccHigh);
  vdReg = _mm_packs_epi32(lowProduct, highProduct);

  /* Write everything out. */
  _mm_store_si128((__m128i*) accLow, vaccLow);
  _mm_store_si128((__m128i*) accMid, vaccMid);
  _mm_store_si128((__m128i*) accHigh, vaccHigh);
  return vdReg;
#else
  debug("Unimplemented function: VMULF.");
#endif
}

/* ============================================================================
 *  Instruction: VMULQ (Vector Multiply MPEG Quantization)
 * ========================================================================= */
__m128i
RSPVMULQ(struct RSPCP2 *unused(cp2), int16_t *unused(vd), __m128i unused(vsReg),
  __m128i unused(vtReg), __m128i unused(vtShuf), __m128i zero) {
  debug("Unimplemented function: VMULQ.");
  return zero;
}

/* ============================================================================
 *  Instruction: VMULU (Vector Multiply of Unsigned Fractions).
 * ========================================================================= */
__m128i
RSPVMULU(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;

#ifdef USE_SSE
  __m128i lowProduct, highProduct, unpackLow, unpackHigh;
  __m128i temp, vaccLow, vaccMid, vaccHigh, vdReg;

  /* Compute 64-bit signed product. */
  unpackLow = _mm_mullo_epi16(vsReg, vtShuf);
  unpackHigh = _mm_mulhi_epi16(vsReg, vtShuf);
  lowProduct = _mm_unpacklo_epi16(unpackLow, unpackHigh);
  highProduct = _mm_unpackhi_epi16(unpackLow, unpackHigh);
  lowProduct = _mm_slli_epi32(lowProduct, 1);
  highProduct = _mm_slli_epi32(highProduct, 1);

  /* Add the rounding value. */
  temp = _mm_set1_epi32(0x00008000U);
  lowProduct = _mm_add_epi32(lowProduct, temp);
  highProduct = _mm_add_epi32(highProduct, temp);

  /* Compute the accmulator, paying attention to overflow. */
  RSPPack32to16(lowProduct, highProduct, &vaccLow, &vaccMid);
  vaccHigh = _mm_cmplt_epi16(vaccMid, zero);
  temp = _mm_cmpeq_epi16(vsReg, vtShuf);
  temp = _mm_cmpeq_epi16(temp, zero);
  vaccHigh = _mm_and_si128(vaccHigh, temp);

  /* Compute the result by clamping the middle word. */
  lowProduct = _mm_unpacklo_epi16(vaccMid, vaccHigh);
  highProduct = _mm_unpackhi_epi16(vaccMid, vaccHigh);
  vdReg = _mm_packs_epi32(lowProduct, highProduct);
  temp = _mm_cmpgt_epi16(vdReg, zero);
  vdReg = _mm_and_si128(vdReg, temp);

  /* Write everything out. */
  _mm_store_si128((__m128i*) accLow, vaccLow);
  _mm_store_si128((__m128i*) accMid, vaccMid);
  _mm_store_si128((__m128i*) accHigh, vaccHigh);
  return vdReg;
#else
  debug("Unimplemented function: VMULU.");
#endif
}

/* ============================================================================
 *  Instruction: VNAND (Vector NAND of Short Elements)
 * ========================================================================= */
__m128i
RSPVNAND(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i unused(zero)) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vdReg;

  vdReg = _mm_nand_si128(vtShuf, vsReg);
  _mm_store_si128((__m128i*) accLow, vdReg);
  return vdReg;
#else
#warning "Unimplemented function: RSPVNAND (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VNE (Vector Select Not Equal)
 * ========================================================================= */
__m128i
RSPVNE(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vne = _mm_load_si128((__m128i*) (cp2->vcohi.slices));
  __m128i notequal = _mm_cmpeq_epi16(vtShuf, vsReg);
  notequal = _mm_cmpeq_epi16(notequal, zero);
  __m128i vvcc = _mm_or_si128(notequal, vne);

  _mm_store_si128((__m128i*) accLow, vsReg);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), zero);
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), zero);
  _mm_store_si128((__m128i*) (cp2->vcclo.slices), vvcc);
  _mm_store_si128((__m128i*) (cp2->vcchi.slices), zero);
  return vsReg;
#else
#warning "Unimplemented function: RSPVNE (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VNOP (Vector No Operation)
 * ========================================================================= */
__m128i
RSPVNOP(struct RSPCP2 *unused(cp2), int16_t *vd, __m128i unused(vsReg),
  __m128i unused(vtReg), __m128i unused(vtShuf), __m128i unused(zero)) {
  return _mm_load_si128((__m128i*) vd);
}

/* ============================================================================
 *  Instruction: VNOR (Vector NOR of Short Elements)
 * ========================================================================= */
__m128i
RSPVNOR(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i unused(zero)) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vdReg;

  vdReg = _mm_nor_si128(vtShuf, vsReg);
  _mm_store_si128((__m128i*) accLow, vdReg);
  return vdReg;
#else
#warning "Unimplemented function: RSPVNOR (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VOR (Vector OR of Short Elements)
 * ========================================================================= */
__m128i
RSPVOR(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i unused(zero)) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vdReg;

  vdReg = _mm_or_si128(vtShuf, vsReg);
  _mm_store_si128((__m128i*) accLow, vdReg);
  return vdReg;
#else
#warning "Unimplemented function: RSPVOR (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VNXOR (Vector NXOR of Short Elements)
 * ========================================================================= */
__m128i
RSPVNXOR(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i unused(zero)) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vdReg;

  vdReg = _mm_nxor_si128(vtShuf, vsReg);
  _mm_store_si128((__m128i*) accLow, vdReg);
  return vdReg;
#else
#warning "Unimplemented function: RSPVNXOR (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VRCP (Vector Element Scalar Reciprocal (Single Precision))
 * ========================================================================= */
__m128i
RSPVRCP(struct RSPCP2 *cp2, int16_t *vd,
  __m128i unused(vsReg), __m128i vtReg, __m128i vtShuf, __m128i unused(zero)) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  unsigned delement = cp2->iw >> 11 & 0x7;
  unsigned element = cp2->iw >> 21 & 0x7;

  int shift;
  unsigned int addr;
  int32_t fetch;
  int32_t data;

  _mm_store_si128((__m128i*) accLow, vtReg);
  if ((data = cp2->divIn = accLow[element]) < 0)
    data = -data;

  if (!data)
    shift = 0x10;
  else
    shift =__builtin_clz(data);

  addr = ((data << shift) & 0x7FC00000) >> 22;
  fetch = (uint32_t) ReciprocalLUT[addr] << 14;
  cp2->divOut = (0x40000000 | fetch) >> (shift ^ 31);

  cp2->divOut ^= cp2->divIn >> 31;
  if (cp2->divIn == 0)
    cp2->divOut = 0x7FFFFFFF;
  else if (cp2->divIn == (int32_t) 0xFFFF8000)
    cp2->divOut = 0xFFFF0000;

#ifdef USE_SSE
  _mm_store_si128((__m128i*) accLow, vtShuf);
#else
#warning "Unimplemented function: RSPVRCP (No SSE)."
#endif

  vd[delement] = (short) cp2->divOut;
  cp2->doublePrecision = 0;
  return _mm_load_si128((__m128i*) vd);
}

/* ============================================================================
 *  Instruction: VRCPH (Vector Element Scalar Reciprocal (Double Prec. High))
 * ========================================================================= */
__m128i
RSPVRCPH(struct RSPCP2 *cp2, int16_t *vd,
  __m128i unused(vsReg), __m128i vtReg, __m128i vtShuf, __m128i unused(zero)) {
  unsigned delement = cp2->iw >> 11 & 0x1F;
  unsigned element = cp2->iw >> 21 & 0xF;
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t vtData[8];

  _mm_storeu_si128((__m128i*) vtData, vtReg);
  cp2->divIn = vtData[element & 0x7] << 16;

#ifdef USE_SSE
  _mm_store_si128((__m128i*) accLow, vtShuf);
#else
#warning "Unimplemented function: RSPVRCPH (No SSE)."
#endif

  vd[delement & 0x7] = cp2->divOut >> 16;
  cp2->doublePrecision = true;
  return _mm_load_si128((__m128i*) vd);
}

/* ============================================================================
 *  Instruction: VRCPL (Vector Element Scalar Reciprocal (Double Prec. Low))
 * ========================================================================= */
__m128i
RSPVRCPL(struct RSPCP2 *cp2, int16_t *vd,
  __m128i unused(vsReg), __m128i vtReg, __m128i vtShuf, __m128i unused(zero)) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  unsigned delement = cp2->iw >> 11 & 0x7;
  unsigned element = cp2->iw >> 21 & 0x7;
  int data, fetch, shift = 32;
  unsigned addr;

  _mm_store_si128((__m128i*) accLow, vtReg);

  if (cp2->doublePrecision) {
    cp2->divIn |= (uint16_t) accLow[element];
    data = cp2->divIn;
    shift = 0x00;

    if (data < 0) {
      if (data >= -32768)
        data = -data;
      else
        data = ~data;
    }
  }

  else {
    data = cp2->divIn = accLow[element];
    shift = 0x10;

    if (data < 0)
      data = -data;
  }

  if (data != 0)
    shift =__builtin_clz(data);

  addr = (data << shift) >> 22;
  fetch = ReciprocalLUT[addr &= 0x000001FF];
  shift ^= 31;
  cp2->divOut = (0x40000000 | (fetch << 14)) >> shift;

  cp2->divOut ^= cp2->divIn >> 31;
  if (cp2->divIn == 0)
    cp2->divOut = 0x7FFFFFFF;
  else if (cp2->divIn == (int32_t) 0xFFFF8000)
    cp2->divOut = 0xFFFF0000;


#ifdef USE_SSE
  _mm_store_si128((__m128i*) accLow, vtShuf);
#else
#warning "Unimplemented function: RSPVRCPL (No SSE)."
#endif

  vd[delement] = (short) cp2->divOut;
  cp2->doublePrecision = false;
  return _mm_load_si128((__m128i*) vd);
}

/* ============================================================================
 *  Instruction: VRNDN (Vector Accumulator DCT Rounding (Negative))
 * ========================================================================= */
__m128i
RSPVRNDN(struct RSPCP2 *unused(cp2), int16_t *unused(vd), __m128i unused(vsReg),
  __m128i unused(vtReg), __m128i unused(vtShuf), __m128i zero) {
  debug("Unimplemented function: VRNDN.");
  return zero;
}

/* ============================================================================
 *  Instruction: VRNDP (Vector Accumulator DCT Rounding (Positive))
 * ========================================================================= */
__m128i
RSPVRNDP(struct RSPCP2 *unused(cp2), int16_t *unused(vd), __m128i unused(vsReg),
  __m128i unused(vtReg), __m128i unused(vtShuf), __m128i zero) {
  debug("Unimplemented function: VRNDP.");
  return zero;
}

/* ============================================================================
 *  Instruction: VRSQ (Vector Element Scalar SQRT Reciprocal)
 * ========================================================================= */
__m128i
RSPVRSQ(struct RSPCP2 *unused(cp2), int16_t *unused(vd), __m128i unused(vsReg),
  __m128i unused(vtReg), __m128i unused(vtShuf), __m128i zero) {
  debug("Unimplemented function: VRSQ.");
  return zero;
}

/* ============================================================================
 *  Instruction: VRSQH (Vector Element Scalar SQRT Reciprocal (Double Prec. H))
 * ========================================================================= */
__m128i
RSPVRSQH(struct RSPCP2 *cp2, int16_t *vd,
  __m128i unused(vsReg), __m128i vtReg, __m128i vtShuf, __m128i unused(zero)) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  unsigned delement = cp2->iw >> 11 & 0x1F;
  unsigned element = cp2->iw >> 21 & 0xF;
  int16_t vtData[8];

  _mm_storeu_si128((__m128i*) vtData, vtReg);
  cp2->divIn = vtData[element & 0x7] << 16;

#ifdef USE_SSE
  _mm_store_si128((__m128i*) accLow, vtShuf);
#else
#warning "Unimplemented function: RSPVRSQH (No SSE)."
#endif

  vd[delement & 07] = cp2->divOut >> 16;
  cp2->doublePrecision = true;
  return _mm_load_si128((__m128i*) vd);
}

/* ============================================================================
 *  Instruction: VRSQL (Vector Element Scalar SQRT Reciprocal (Double Prec. L))
 * ========================================================================= */
__m128i
RSPVRSQL(struct RSPCP2 *cp2, int16_t *vd,
  __m128i unused(vsReg), __m128i vtReg, __m128i vtShuf, __m128i unused(zero)) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  unsigned delement = cp2->iw >> 11 & 0x7;
  unsigned element = cp2->iw >> 21 & 0x7;

  int32_t data, fetch;
  unsigned addr;
  int shift;

  _mm_store_si128((__m128i*) accLow, vtReg);

  if (cp2->doublePrecision) {
    cp2->divIn |= (uint16_t) accLow[element];
    data = cp2->divIn;
    shift = 0x00;

    if (data < 0) {
      if (data >= -32768)
        data = -data;
      else
        data = ~data;
    }
  }

  else {
    data = cp2->divIn = accLow[element];
    shift = 0x10;

    if (data < 0)
      data = -data;
  }

  if (data != 0)
    shift =__builtin_clz(data);

  addr = (data << shift) >> 22;
  addr = ((addr | 0x200) & 0x3FE) | (shift & 1);
  fetch = ReciprocalLUT[addr];

  shift = (shift ^ 31) >> 1;
  cp2->divOut = (0x40000000 | (fetch << 14)) >> shift;

  cp2->divOut ^= cp2->divIn >> 31;
  if (cp2->divIn == 0)
    cp2->divOut = 0x7FFFFFFF;
  else if (cp2->divIn == (int32_t) 0xFFFF8000)
    cp2->divOut = 0xFFFF0000;

#ifdef USE_SSE
  _mm_store_si128((__m128i*) accLow, vtShuf);
#else
#warning "Unimplemented function: RSPVRSQL (No SSE)."
#endif

  vd[delement & 0x7] = (short) cp2->divOut;
  cp2->doublePrecision = false;
  return _mm_load_si128((__m128i*) vd);
}

/* ============================================================================
 *  Instruction: VSAR (Vector Accumulator Read (and Write))
 * ========================================================================= */
__m128i
RSPVSAR(struct RSPCP2 *cp2, int16_t *unused(vd), __m128i unused(vsReg),
  __m128i unused(vtReg), __m128i unused(vtShuf), __m128i zero) {
  unsigned element = cp2->iw >> 21 & 0x7;
  __m128i vector = zero;

  if (element < 3)
    vector = _mm_load_si128((__m128i*) (&cp2->accumulatorHigh + element));

  return vector;
}

/* ============================================================================
 *  Instruction: VSUB (Vector Subtraction of Short Elements)
 * ========================================================================= */
__m128i
RSPVSUB(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vtRegPos, vtRegNeg, vaccLow, vdReg;
  __m128i unsatDiff, vMask, carryOut;

  carryOut = _mm_load_si128((__m128i*) (cp2->vcolo.slices));
  carryOut = _mm_srli_epi16(carryOut, 15);

  /* VACC uses unsaturated arithmetic. */
  vaccLow = unsatDiff = _mm_sub_epi16(vsReg, vtShuf);
  vaccLow = _mm_sub_epi16(vaccLow, carryOut);

  /* VD is the signed diff of the two sources and the carry. Since we */
  /* have to saturate the diff of all three, we have to be clever. */
  vtRegNeg = _mm_cmplt_epi16(vtShuf, zero);
  vtRegPos = _mm_cmpeq_epi16(vtRegNeg, zero);

  vdReg = _mm_subs_epi16(vsReg, vtShuf);
  vMask = _mm_cmpeq_epi16(unsatDiff, vdReg);
  vMask = _mm_and_si128(vtRegNeg, vMask);

  vtRegNeg = _mm_and_si128(vMask, carryOut);
  vtRegPos = _mm_and_si128(vtRegPos, carryOut);
  carryOut = _mm_or_si128(vtRegNeg, vtRegPos);
  vdReg = _mm_subs_epi16(vdReg, carryOut);

  _mm_store_si128((__m128i*) accLow, vaccLow);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), zero);
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), zero);
  return vdReg;
#else
#warning "Unimplemented function: RSPVSUB (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VSUBC (Vector Subtraction of Short Elements with Carry)
 * ========================================================================= */
__m128i
RSPVSUBC(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i zero) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i satDiff, lessThanMask, notEqualMask, vdReg;

  satDiff = _mm_subs_epu16(vsReg, vtShuf);
  vdReg = _mm_sub_epi16(vsReg, vtShuf);

  /* Set the carry out flags when difference is < 0. */
  notEqualMask = _mm_cmpeq_epi16(vdReg, zero);
  notEqualMask = _mm_cmpeq_epi16(notEqualMask, zero);
  lessThanMask = _mm_cmpeq_epi16(satDiff, zero);
  lessThanMask = _mm_and_si128(lessThanMask, notEqualMask);

  _mm_store_si128((__m128i*) accLow, vdReg);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), lessThanMask);
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), notEqualMask);
  return vdReg;
#else
#warning "Unimplemented function: RSPVSUBC (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VXOR (Vector XOR of Short Elements)
 * ========================================================================= */
__m128i
RSPVXOR(struct RSPCP2 *cp2, int16_t *unused(vd),
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf, __m128i unused(zero)) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vdReg;

  vdReg = _mm_xor_si128(vtShuf, vsReg);
  _mm_store_si128((__m128i*) accLow, vdReg);
  return vdReg;
#else
#warning "Unimplemented function: RSPVXOR (No SSE)."
#endif
}

/* ============================================================================
 *  RSPInitCP2: Initializes the co-processor.
 * ========================================================================= */
void
RSPInitCP2(struct RSPCP2 *cp2) {
  debug("Initializing CP2.");
  memset(cp2, 0, sizeof(*cp2));
}

/* ============================================================================
 *  RSPCycleCP2: Vector execute/multiply/accumulate stages.
 * ========================================================================= */
void
RSPCycleCP2(struct RSPCP2 *cp2) {
  cp2->locked[cp2->accStageDest] = false;     /* "WB" */
  cp2->accStageDest = cp2->mulStageDest;      /* "DF" */
  cp2->mulStageDest = 0;

  if (cp2->opcode.id != RSP_OPCODE_VINV) {
    uint32_t iw = cp2->iw;
    unsigned vtRegister = iw >> 16 & 0x1F;
    unsigned vsRegister = iw >> 11 & 0x1F;
    unsigned vdRegister = iw >> 6 & 0x1F;

    int16_t *vd = cp2->regs[vdRegister].slices;
    __m128i vt = _mm_load_si128((__m128i*) (cp2->regs[vtRegister].slices));
    __m128i vs = _mm_load_si128((__m128i*) (cp2->regs[vsRegister].slices));
    __m128i vtShuf = RSPGetVectorOperands(vt, iw >> 21 & 0xF);
    __m128i zero = _mm_setzero_si128();

    __m128i vdReg = RSPVectorFunctionTable[cp2->opcode.id](
      cp2, vd, vs, vt, vtShuf, zero);

    _mm_store_si128((__m128i*) vd, vdReg);
    cp2->mulStageDest = (vd - cp2->regs[0].slices) >> 3;
  }

#ifndef NDEBUG
  cp2->counts[cp2->opcode.id]++;
#endif
}

/* ============================================================================
 *  RSPCP2GetAccumulator: Fetches an accumulator and returns it.
 * ========================================================================= */
#ifndef NDEBUG
void
RSPCP2GetAccumulator(const struct RSPCP2 *cp2, unsigned reg, uint16_t *acc) {
#ifdef USE_SSE
  acc[0] = cp2->accumulatorLow.slices[reg];
  acc[1] = cp2->accumulatorMid.slices[reg];
  acc[2] = cp2->accumulatorHigh.slices[reg];
#else
#warning "Unimplemented function: RSPCP2GetAccumulator (No SSE)."
#endif
}
#endif

/* ============================================================================
 *  RSPCP2GetCarryOut: Fetches the carry-out and returns it.
 * ========================================================================= */
#ifndef NDEBUG
uint16_t
RSPCP2GetCarryOut(const struct RSPCP2 *cp2) {
#ifdef USE_SSE
  __m128i carryOut = _mm_load_si128((__m128i*) cp2->carryOut.slices);
  return _mm_movemask_epi8(carryOut);
#else
#warning "Unimplemented function: RSPCP2GetCarryOut (No SSE)."
  return 0;
#endif
}
#endif

