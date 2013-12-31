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
RSPClampLowToVal(__m128i vaccLow, __m128i vaccMid, __m128i vaccHigh) {
  __m128i midSign, negCheck, negVal, posVal, useValMask, zero;

  /* Compute some common values ahead of time. */
  zero = _mm_xor_si128(vaccLow, vaccLow);
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
RSPPackLo32to16(__m128i vectorLow, __m128i vectorHigh) {
#ifdef SSSE3_ONLY
  vectorLow = _mm_slli_epi32(vectorLow, 16);
  vectorHigh = _mm_slli_epi32(vectorHigh, 16);
  vectorLow = _mm_srai_epi32(vectorLow, 16);
  vectorHigh = _mm_srai_epi32(vectorHigh, 16);
  return _mm_packs_epi32(vectorLow, vectorHigh);
#else
  vectorLow = _mm_blend_epi16(vectorLow, _mm_setzero_si128(), 0xAA);
  vectorHigh = _mm_blend_epi16(vectorHigh, _mm_setzero_si128(), 0xAA);
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
RSPZeroExtend16to32(__m128i source, __m128i *vectorLow, __m128i *vectorHigh) {
  *vectorHigh = _mm_unpackhi_epi16(source, _mm_setzero_si128());
  *vectorLow = _mm_unpacklo_epi16(source, _mm_setzero_si128());
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
void
RSPVABS(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i valLessThan, signLessThan, resultLessThan, vdReg;
  vdReg = _mm_sign_epi16(vtShuf, vsReg);

  /* _mm_sign_epi16 will not fixup INT16_MIN; the RSP will! */
  resultLessThan = _mm_cmplt_epi16(vdReg, _mm_setzero_si128());
  signLessThan = _mm_cmplt_epi16(vsReg, _mm_setzero_si128());
  valLessThan = _mm_cmplt_epi16(vtShuf, _mm_setzero_si128());

  valLessThan = _mm_and_si128(valLessThan, signLessThan);
  resultLessThan = _mm_and_si128(valLessThan, resultLessThan);
  vdReg = _mm_xor_si128(vdReg, resultLessThan);

  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) accLow, vdReg);
#else
#warning "Unimplemented function: RSPVABS (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VADD (Vector Add of Short Elements)
 * ========================================================================= */
void
RSPVADD(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
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

  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) accLow, vaccLow);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), _mm_setzero_si128());
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), _mm_setzero_si128());
#else
#warning "Unimplemented function: RSPVADD (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VADDC (Vector Add of Short Elements with Carry)
 * ========================================================================= */
void
RSPVADDC(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i satSum, unsatSum, equalMask;
  satSum = _mm_adds_epu16(vsReg, vtShuf);
  unsatSum = _mm_add_epi16(vsReg, vtShuf);

  equalMask = _mm_cmpeq_epi16(satSum, unsatSum);
  equalMask = _mm_cmpeq_epi16(equalMask, _mm_setzero_si128());

  _mm_store_si128((__m128i*) vd, unsatSum);
  _mm_store_si128((__m128i*) accLow, unsatSum);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), equalMask);
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), _mm_setzero_si128());
#else
#warning "Unimplemented function: RSPVADDC (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VAND (Vector AND of Short Elements)
 * ========================================================================= */
void
RSPVAND(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vdReg;

  vdReg = _mm_and_si128(vtShuf, vsReg);

  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) accLow, vdReg);
#else
#warning "Unimplemented function: RSPVAND (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VCH (Vector Select Clip Test High)
 * ========================================================================= */
void
RSPVCH(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i cmp1, cmp2, negVtReg, snAluOp, temp, temp2a, temp2s;
  __m128i ge, le, neq, sn, zero;

  /* sn = (vs ^ vt) < 0 */
  zero = _mm_xor_si128(vsReg, vsReg);
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
  temp = _mm_packs_epi16(temp, temp);
  cp2->vce = _mm_movemask_epi8(temp);

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
  _mm_store_si128((__m128i*) vd, temp);
#else
#warning "Unimplemented function: RSPVCH (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VCL (Vector Select Clip Test Low)
 * ========================================================================= */
void
RSPVCL(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;

  uint16_t vco = RSPGetVCO(cp2);
  int16_t vccOld = RSPGetVCC(cp2);
  int16_t vtData[8], vsData[8];
  int16_t vcc = 0;
  int ge, le;
  unsigned i;

#ifdef USE_SSE
  _mm_storeu_si128((__m128i*) vtData, vtShuf);
  _mm_storeu_si128((__m128i*) vsData, vsReg);
#else
#warning "Unimplemented function: RSPVCL (No SSE)."
#endif

  for (i = 0; i < 8; i++) {
    uint16_t vs = (unsigned short) vsData[i];
    uint16_t vt = (unsigned short) vtData[i];
    int eq = (~vco >> (i + 0x8)) & 0x0001;
    int sn = (vco >> (i + 0x0)) & 0x0001;

    le = vccOld & (0x0001 << i);
    ge = vccOld & (0x0100 << i);

    if (sn) {
      if (eq) {
        int sum = vs + vt;
        int ce = (cp2->vce >> i) & 0x01;
        int lz = ((sum & 0x0000FFFF) == 0x00000000);
        int uz = ((sum & 0xFFFF0000) == 0x00000000);

        le = ((!ce) & (lz & uz)) | (ce & (lz | uz));
        le <<= i + 0x0;
      }

      accLow[i] = le ? -vt : vs;
    }

    else {
      if (eq) {
        ge = (vs - vt >= 0);
        ge <<= i + 0x8;
      }

      accLow[i] = ge ? vt : vs;
    }

    vcc |= ge | le;
  }

  RSPSetVCC(cp2, vcc);
  memcpy(vd, accLow, sizeof(short) * 8);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), _mm_setzero_si128());
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), _mm_setzero_si128());
  cp2->vce = 0x00;
}

/* ============================================================================
 *  Instruction: VCR (Vector Select Crimp Test Low)
 * ========================================================================= */
void
RSPVCR(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i cmp1, cmp2, negVtReg, notVtReg, snAluOp, temp, temp2a, temp2s;
  __m128i ge, le, sn, zero;

  /* sn = (vs ^ vt) < 0 */
  zero = _mm_xor_si128(vsReg, vsReg);
  sn = _mm_xor_si128(vsReg, vtShuf);
  sn = _mm_cmplt_epi16(sn, zero);

  /* if ( sn) { snAluOp = (vs + vt); } */
  /* if (!sn) { snAluOp = (vs - vt); } */
  notVtReg = _mm_xor_si128(vtShuf, sn);
  negVtReg = _mm_add_epi16(notVtReg, sn);
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
  _mm_store_si128((__m128i*) vd, temp);

  temp = _mm_packs_epi16(le, ge);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), _mm_setzero_si128());
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), _mm_setzero_si128());
  _mm_store_si128((__m128i*) (cp2->vcclo.slices), le);
  _mm_store_si128((__m128i*) (cp2->vcchi.slices), ge);
  cp2->vce = 0x00;
#else
#warning "Unimplemented function: RSPVCR (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VEQ (Vector Select Equal)
 * ========================================================================= */
void
RSPVEQ(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vne = _mm_load_si128((__m128i*) (cp2->vcohi.slices));
  __m128i equal = _mm_cmpeq_epi16(vtShuf, vsReg);
  vne = _mm_cmpeq_epi16(vne, _mm_setzero_si128());
  __m128i vvcc = _mm_and_si128(equal, vne);

  _mm_store_si128((__m128i*) accLow, vtShuf);
  _mm_store_si128((__m128i*) vd, vtShuf);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), _mm_setzero_si128());
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), _mm_setzero_si128());
  _mm_store_si128((__m128i*) (cp2->vcclo.slices), vvcc);
  _mm_store_si128((__m128i*) (cp2->vcchi.slices), _mm_setzero_si128());
#else
#warning "Unimplemented function: RSPVEQ (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VGE (Vector Select Greater Than or Equal)
 * ========================================================================= */
void
RSPVGE(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vne = _mm_load_si128((__m128i*) (cp2->vcohi.slices));
  __m128i vco = _mm_load_si128((__m128i*) (cp2->vcolo.slices));
  __m128i temp, equal, greaterEqual, vdReg;

  /* equal = (~vco | ~vne) && (vs == vt) */
  temp = _mm_and_si128(vne, vco);
  temp = _mm_cmpeq_epi16(temp, _mm_setzero_si128());
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
  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), _mm_setzero_si128());
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), _mm_setzero_si128());
  _mm_store_si128((__m128i*) (cp2->vcclo.slices), greaterEqual);
  _mm_store_si128((__m128i*) (cp2->vcchi.slices), _mm_setzero_si128());
#else
#warning "Unimplemented function: RSPVGE (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VINV (Invalid Vector Operation)
 * ========================================================================= */
void
RSPVINV(struct RSPCP2 *unused(cp2), int16_t *unused(vd),
  __m128i unused(vsReg), __m128i unused(vtReg), __m128i unused(vtShuf)) {
}

/* ============================================================================
 *  Instruction: VLT (Vector Select Less Than or Equal)
 * ========================================================================= */
void
RSPVLT(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
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
  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), _mm_setzero_si128());
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), _mm_setzero_si128());
  _mm_store_si128((__m128i*) (cp2->vcclo.slices), lessthanEqual);
  _mm_store_si128((__m128i*) (cp2->vcchi.slices), _mm_setzero_si128());
#else
#warning "Unimplemented function: RSPVLT (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VMACF (Vector Multiply-Accumulate of Signed Fractions)
 * ========================================================================= */
void
RSPVMACF(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
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
  RSPZeroExtend16to32(vaccLow, &vaccLow, &vaccHigh);

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
  vdRegLo = _mm_blend_epi16(loProduct, _mm_setzero_si128(), 0xAA);
  vdRegHi = _mm_blend_epi16(hiProduct, _mm_setzero_si128(), 0xAA);
#endif

  vaccLow = _mm_add_epi32(vaccLow, vdRegLo);
  vaccHigh = _mm_add_epi32(vaccHigh, vdRegHi);

  vdReg = RSPPackLo32to16(vaccLow, vaccHigh);
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

  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) accMid, vaccMid);
  _mm_store_si128((__m128i*) accHigh, vaccHigh);
#else
#warning "Unimplemented function: VMACF (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VMACQ (Vector Accumulator Oddification)
 * ========================================================================= */
void
RSPVMACQ(struct RSPCP2 *unused(cp2), int16_t *unused(vd),
  __m128i unused(vsReg), __m128i unused(vtReg), __m128i unused(vtShuf)) {
  debug("Unimplemented function: VMACQ.");
}

/* ============================================================================
 *  Instruction: VMACU (Vector Multiply-Accumulate of Unsigned Fractions)
 * ========================================================================= */
void
RSPVMACU(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
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
  RSPZeroExtend16to32(vaccLow, &vaccLow, &vaccHigh);

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
  vdRegLo = _mm_blend_epi16(loProduct, _mm_setzero_si128(), 0xAA);
  vdRegHi = _mm_blend_epi16(hiProduct, _mm_setzero_si128(), 0xAA);
#endif

  vaccLow = _mm_add_epi32(vaccLow, vdRegLo);
  vaccHigh = _mm_add_epi32(vaccHigh, vdRegHi);

  vdReg = RSPPackLo32to16(vaccLow, vaccHigh);
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
}

/* ============================================================================
 *  Instruction: VMADH (Vector Multiply-Accumulate of High Partial Products)
 * ========================================================================= */
void
RSPVMADH(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
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

  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) accMid, vaccMid);
  _mm_store_si128((__m128i*) accHigh, vaccHigh);
#else
#warning "Unimplemented function: RSPVMADH (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VMADL (Vector Multiply-Accumulate of Lower Partial Products).
 * ========================================================================= */
void
RSPVMADL(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
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
  RSPZeroExtend16to32(vaccLow, &vaccLow, &vaccHigh);

  /* Begin accumulating the products. */
  unpackHi = _mm_mulhi_epu16(vsReg, vtShuf);
  loProduct = _mm_unpacklo_epi16(unpackHi, _mm_setzero_si128());
  hiProduct = _mm_unpackhi_epi16(unpackHi, _mm_setzero_si128());

  vaccLow = _mm_add_epi32(vaccLow, loProduct);
  vaccHigh = _mm_add_epi32(vaccHigh, hiProduct);
  vdReg = RSPPackLo32to16(vaccLow, vaccHigh);
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
  vdReg = RSPClampLowToVal(vdReg, vaccMid, vaccHigh);

  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) accMid, vaccMid);
  _mm_store_si128((__m128i*) accHigh, vaccHigh);
#else
#warning "Unimplemented function: RSPVMADL (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VMADM (Vector Multiply-Accumulate of Mid Partial Products)
 * ========================================================================= */
void
RSPVMADM(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
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
  RSPZeroExtend16to32(vtShuf, &vtRegLo, &vtRegHi);
  RSPZeroExtend16to32(vaccLow, &vaccLow, &vaccHigh);

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
  vdRegLo = _mm_blend_epi16(loProduct, _mm_setzero_si128(), 0xAA);
  vdRegHi = _mm_blend_epi16(hiProduct, _mm_setzero_si128(), 0xAA);
#endif
  vaccLow = _mm_add_epi32(vaccLow, vdRegLo);
  vaccHigh = _mm_add_epi32(vaccHigh, vdRegHi);

  vdReg = RSPPackLo32to16(vaccLow, vaccHigh);
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

  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) accMid, vaccMid);
  _mm_store_si128((__m128i*) accHigh, vaccHigh);
#else
#warning "Unimplemented function: RSPVMADM (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VMADN (Vector Multiply-Accumulate of Mid Partial Products)
 * ========================================================================= */
void
RSPVMADN(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;

#ifdef USE_SSE
  __m128i vaccTemp, vaccLow, vaccMid, vaccHigh, loProduct, hiProduct;
  __m128i vsRegLo, vsRegHi, vtRegLo, vtRegHi, vdReg, vdRegLo, vdRegHi;
  vaccLow = _mm_load_si128((__m128i*) accLow);
  vaccMid = _mm_load_si128((__m128i*) accMid);

  /* Unpack to obtain for 32-bit precision. */
  RSPZeroExtend16to32(vsReg, &vsRegLo, &vsRegHi);
  RSPSignExtend16to32(vtShuf, &vtRegLo, &vtRegHi);
  RSPZeroExtend16to32(vaccLow, &vaccLow, &vaccHigh);

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
  vdRegLo = _mm_blend_epi16(loProduct, _mm_setzero_si128(), 0xAA);
  vdRegHi = _mm_blend_epi16(hiProduct, _mm_setzero_si128(), 0xAA);
#endif

  vaccLow = _mm_add_epi32(vaccLow, vdRegLo);
  vaccHigh = _mm_add_epi32(vaccHigh, vdRegHi);

   vdReg = RSPPackLo32to16(vaccLow, vaccHigh);
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
  vdReg = RSPClampLowToVal(vdReg, vaccMid, vaccHigh);

  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) accMid, vaccMid);
  _mm_store_si128((__m128i*) accHigh, vaccHigh);
#else
#warning "Unimplemented function: RSPVMADN (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VMOV (Vector Element Scalar Move)
 * ========================================================================= */
void
RSPVMOV(struct RSPCP2 *cp2, int16_t *vd,
  __m128i unused(vsReg), __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  unsigned delement = cp2->iw >> 11 & 0x1F;

#ifdef USE_SSE
  _mm_store_si128((__m128i*) accLow, vtShuf);
#else
#warning "Unimplemented function: RSPVMOV (No SSE)."
#endif

  vd[delement & 0x7] = accLow[delement & 0x7];
}

/* ============================================================================
 *  Instruction: VMRG (Vector Select Merge)
 * ========================================================================= */
void
RSPVMRG(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
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
  _mm_store_si128((__m128i*) vd, vaccLow);
#else
#warning "Unimplemented function: RSPVMRG (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VMUDH (Vector Multiply of High Partial Products)
 * ========================================================================= */
void
RSPVMUDH(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
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

  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) accLow, _mm_setzero_si128());
  _mm_store_si128((__m128i*) accMid, vaccMid);
  _mm_store_si128((__m128i*) accHigh, vaccHigh);
#else
  debug("Unimplemented function: VMUDH.");
#endif
}

/* ============================================================================
 *  Instruction: VMUDL (Vector Multiply of Low Partial Products)
 * ========================================================================= */
void
RSPVMUDL(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vt), __m128i vtShuf) {
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

  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) accLow, vdReg);
  _mm_store_si128((__m128i*) accMid, _mm_setzero_si128());
  _mm_store_si128((__m128i*) accHigh, _mm_setzero_si128());
#else
  debug("Unimplemented function: VMUDL.");
#endif
}

/* ============================================================================
 *  Instruction: VMUDM (Vector Multiply of Middle Partial Products)
 * ========================================================================= */
void
RSPVMUDM(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;

#ifdef USE_SSE
  __m128i vsRegLo, vsRegHi, vtRegLo, vtRegHi, vdReg;
  __m128i loProduct, hiProduct, vaccLow, vaccHigh;

  /* Unpack to obtain for 32-bit precision. */
  RSPSignExtend16to32(vsReg, &vsRegLo, &vsRegHi);
  RSPZeroExtend16to32(vtShuf, &vtRegLo, &vtRegHi);

  /* Begin accumulating the products. */
  loProduct = _mm_mullo_epi32(vsRegLo, vtRegLo);
  hiProduct = _mm_mullo_epi32(vsRegHi, vtRegHi);
  RSPPack32to16(loProduct, hiProduct, &vaccLow, &vaccHigh);

  loProduct = _mm_cmplt_epi32(loProduct, _mm_setzero_si128());
  hiProduct = _mm_cmplt_epi32(hiProduct, _mm_setzero_si128());
  vdReg = _mm_packs_epi32(loProduct, hiProduct);

  _mm_store_si128((__m128i*) vd, vaccHigh);
  _mm_store_si128((__m128i*) accLow, vaccLow);
  _mm_store_si128((__m128i*) accMid, vaccHigh);
  _mm_store_si128((__m128i*) accHigh, vdReg);
#else
  debug("Unimplemented function: VMUDM.");
#endif
}

/* ============================================================================
 *  Instruction: VMUDN (Vector Multiply of Middle Partial Products)
 * ========================================================================= */
void
RSPVMUDN(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;

#ifdef USE_SSE
  __m128i vsRegLo, vsRegHi, vtRegLo, vtRegHi, vdReg;
  __m128i loProduct, hiProduct, vaccLow, vaccHigh;

  /* Unpack to obtain for 32-bit precision. */
  RSPZeroExtend16to32(vsReg, &vsRegLo, &vsRegHi);
  RSPSignExtend16to32(vtShuf, &vtRegLo, &vtRegHi);

  /* Begin accumulating the products. */
  loProduct = _mm_mullo_epi32(vsRegLo, vtRegLo);
  hiProduct = _mm_mullo_epi32(vsRegHi, vtRegHi);
  RSPPack32to16(loProduct, hiProduct, &vaccLow, &vaccHigh);
  vdReg = _mm_cmplt_epi16(vaccHigh, _mm_setzero_si128());

  _mm_store_si128((__m128i*) vd, vaccLow);
  _mm_store_si128((__m128i*) accLow, vaccLow);
  _mm_store_si128((__m128i*) accMid, vaccHigh);
  _mm_store_si128((__m128i*) accHigh, vdReg);
#else
  debug("Unimplemented function: VMUDN.");
#endif
}

/* ============================================================================
 *  Instruction: VMULF (Vector Multiply of Signed Fractions).
 * ========================================================================= */
void
RSPVMULF(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;

#ifdef USE_SSE
  __m128i lowProduct, highProduct, unpackLow, unpackHigh;
  __m128i temp, vaccLow, vaccMid, vaccHigh, vdReg, zero;
  zero = _mm_setzero_si128();

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
  _mm_store_si128((__m128i*) vd, vdReg);
#else
  debug("Unimplemented function: VMULF.");
#endif
}

/* ============================================================================
 *  Instruction: VMULQ (Vector Multiply MPEG Quantization)
 * ========================================================================= */
void
RSPVMULQ(struct RSPCP2 *unused(cp2), int16_t *unused(vd),
  __m128i unused(vsReg), __m128i unused(vtReg), __m128i unused(vtShuf)) {
  debug("Unimplemented function: VMULQ.");
}

/* ============================================================================
 *  Instruction: VMULU (Vector Multiply of Unsigned Fractions).
 * ========================================================================= */
void
RSPVMULU(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;

#ifdef USE_SSE
  __m128i lowProduct, highProduct, unpackLow, unpackHigh;
  __m128i temp, vaccLow, vaccMid, vaccHigh, vdReg, zero;
  zero = _mm_setzero_si128();

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
  _mm_store_si128((__m128i*) vd, vdReg);
#else
  debug("Unimplemented function: VMULU.");
#endif
}

/* ============================================================================
 *  Instruction: VNAND (Vector NAND of Short Elements)
 * ========================================================================= */
void
RSPVNAND(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vdReg;

  vdReg = _mm_nand_si128(vtShuf, vsReg);

  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) accLow, vdReg);
#else
#warning "Unimplemented function: RSPVNAND (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VNE (Vector Select Not Equal)
 * ========================================================================= */
void
RSPVNE(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vne = _mm_load_si128((__m128i*) (cp2->vcohi.slices));
  __m128i notequal = _mm_cmpeq_epi16(vtShuf, vsReg);
  notequal = _mm_cmpeq_epi16(notequal, _mm_setzero_si128());
  __m128i vvcc = _mm_or_si128(notequal, vne);

  _mm_store_si128((__m128i*) accLow, vsReg);
  _mm_store_si128((__m128i*) vd, vsReg);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), _mm_setzero_si128());
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), _mm_setzero_si128());
  _mm_store_si128((__m128i*) (cp2->vcclo.slices), vvcc);
  _mm_store_si128((__m128i*) (cp2->vcchi.slices), _mm_setzero_si128());
#else
#warning "Unimplemented function: RSPVNE (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VNOP (Vector No Operation)
 * ========================================================================= */
void
RSPVNOP(struct RSPCP2 *unused(cp2), int16_t *unused(vd),
  __m128i unused(vsReg), __m128i unused(vtReg), __m128i unused(vtShuf)) {
}

/* ============================================================================
 *  Instruction: VNOR (Vector NOR of Short Elements)
 * ========================================================================= */
void
RSPVNOR(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vdReg;

  vdReg = _mm_nor_si128(vtShuf, vsReg);

  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) accLow, vdReg);
#else
#warning "Unimplemented function: RSPVNOR (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VOR (Vector OR of Short Elements)
 * ========================================================================= */
void
RSPVOR(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vdReg;

  vdReg = _mm_or_si128(vtShuf, vsReg);

  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) accLow, vdReg);
#else
#warning "Unimplemented function: RSPVOR (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VNXOR (Vector NXOR of Short Elements)
 * ========================================================================= */
void
RSPVNXOR(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vdReg;

  vdReg = _mm_nxor_si128(vtShuf, vsReg);

  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) accLow, vdReg);
#else
#warning "Unimplemented function: RSPVNXOR (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VRCP (Vector Element Scalar Reciprocal (Single Precision))
 * ========================================================================= */
void
RSPVRCP(struct RSPCP2 *cp2, int16_t *vd,
  __m128i unused(vsReg), __m128i vtReg, __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  unsigned delement = cp2->iw >> 11 & 0x1F;
  unsigned element = cp2->iw >> 21 & 0xF;
  int16_t vtData[8];

  unsigned int addr;
  int data;
  int fetch;
  int shift = 32;

  _mm_storeu_si128((__m128i*) vtData, vtReg);
  cp2->divIn = (int) vtData[element & 07];
  data = cp2->divIn;

  if (data < 0)
      data = -data;

  /* while (shift > 0) or ((shift ^ 31) < 32) */
  do {
    --shift;
    if (data & (1 << shift))
      goto FOUND_MSB;
  } while (shift);

  shift = 16 ^ 31;
FOUND_MSB:
  shift ^= 31;
  addr = (data << shift) >> 22;
  fetch = ReciprocalLUT[addr &= 0x000001FF];
  shift ^= 31;
  cp2->divOut = (0x40000000 | (fetch << 14)) >> shift;

  if (cp2->divIn < 0)
    cp2->divOut = ~cp2->divOut;
  else if (cp2->divIn == 0)
    cp2->divOut = 0x7FFFFFFF;
  else if (cp2->divIn == -32768)
    cp2->divOut = 0xFFFF0000;

#ifdef USE_SSE
  _mm_store_si128((__m128i*) accLow, vtShuf);
#else
#warning "Unimplemented function: RSPVRCP (No SSE)."
#endif

    vd[delement & 07] = (short) cp2->divOut;
    cp2->doublePrecision = 0;
}

/* ============================================================================
 *  Instruction: VRCPH (Vector Element Scalar Reciprocal (Double Prec. High))
 * ========================================================================= */
void
RSPVRCPH(struct RSPCP2 *cp2, int16_t *vd,
  __m128i unused(vsReg), __m128i vtReg, __m128i vtShuf) {
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
}

/* ============================================================================
 *  Instruction: VRCPL (Vector Element Scalar Reciprocal (Double Prec. Low))
 * ========================================================================= */
void
RSPVRCPL(struct RSPCP2 *cp2, int16_t *vd,
  __m128i unused(vsReg), __m128i vtReg, __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  unsigned delement = cp2->iw >> 11 & 0x1F;
  unsigned element = cp2->iw >> 21 & 0xF;
  int data, fetch, shift = 32;
  int16_t vtData[8];
  unsigned addr;

  _mm_storeu_si128((__m128i*) vtData, vtReg);

  if (cp2->doublePrecision)
    cp2->divIn |= (unsigned short) vtData[element & 0x7];

  else
    cp2->divIn = vtData[element & 0x7] & 0x0000FFFF;

  data = cp2->divIn;
  if (data < 0)
      /* -(x) if >=; ~(x) if < */
      data = -data - (data < -32768);

  /* while (shift > 0) or ((shift ^ 31) < 32) */
  do {
    --shift;

    if (data & (1 << shift))
      goto FOUND_MSB;
  } while (shift);

  /* if (data == 0) shift = DPH ? 16 ^ 31 : 0 ^ 31; */
  shift = 31 - 16 * (int) cp2->doublePrecision;

FOUND_MSB:
  shift ^= 31;
  addr = (data << shift) >> 22;
  fetch = ReciprocalLUT[addr &= 0x000001FF];
  shift ^= 31;
  cp2->divOut = (0x40000000 | (fetch << 14)) >> shift;

  if (cp2->divIn < 0)
    cp2->divOut = ~cp2->divOut;
  else if (cp2->divIn == 0)
    cp2->divOut = 0x7FFFFFFF;
  else if (cp2->divIn == -32768)
    cp2->divOut = 0xFFFF0000;

#ifdef USE_SSE
  _mm_store_si128((__m128i*) accLow, vtShuf);
#else
#warning "Unimplemented function: RSPVRCPL (No SSE)."
#endif

  vd[delement & 0x7] = (short) cp2->divOut;
  cp2->doublePrecision = false;
}

/* ============================================================================
 *  Instruction: VRNDN (Vector Accumulator DCT Rounding (Negative))
 * ========================================================================= */
void
RSPVRNDN(struct RSPCP2 *unused(cp2), int16_t *unused(vd),
  __m128i unused(vsReg), __m128i unused(vtReg), __m128i unused(vtShuf)) {
  debug("Unimplemented function: VRNDN.");
}

/* ============================================================================
 *  Instruction: VRNDP (Vector Accumulator DCT Rounding (Positive))
 * ========================================================================= */
void
RSPVRNDP(struct RSPCP2 *unused(cp2), int16_t *unused(vd),
  __m128i unused(vsReg), __m128i unused(vtReg), __m128i unused(vtShuf)) {
  debug("Unimplemented function: VRNDP.");
}

/* ============================================================================
 *  Instruction: VRSQ (Vector Element Scalar SQRT Reciprocal)
 * ========================================================================= */
void
RSPVRSQ(struct RSPCP2 *unused(cp2), int16_t *unused(vd),
  __m128i unused(vsReg), __m128i unused(vtReg), __m128i unused(vtShuf)) {
  debug("Unimplemented function: VRSQ.");
}

/* ============================================================================
 *  Instruction: VRSQH (Vector Element Scalar SQRT Reciprocal (Double Prec. H))
 * ========================================================================= */
void
RSPVRSQH(struct RSPCP2 *cp2, int16_t *vd,
  __m128i unused(vsReg), __m128i vtReg, __m128i vtShuf) {
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
}

/* ============================================================================
 *  Instruction: VRSQL (Vector Element Scalar SQRT Reciprocal (Double Prec. L))
 * ========================================================================= */
void
RSPVRSQL(struct RSPCP2 *cp2, int16_t *vd,
  __m128i unused(vsReg), __m128i vtReg, __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  unsigned delement = cp2->iw >> 11 & 0x1F;
  unsigned element = cp2->iw >> 21 & 0xF;
  int data, fetch, shift = 32;
  int16_t vtData[8];
  unsigned addr;

  _mm_storeu_si128((__m128i*) vtData, vtReg);

  if (cp2->doublePrecision)
    cp2->divIn |= (unsigned short) vtData[element & 0x7];
  else
    cp2->divIn  = vtData[element & 0x7] & 0x0000FFFF; /* Do not sign-extend. */

  data = cp2->divIn;

  if (data < 0)
    /* -(x) if >=; ~(x) if < */
    data = -data - (data < -32768);

  /* while (shift > 0) or ((shift ^ 31) < 32) */
  do {
    --shift;
    if (data & (1 << shift))
      goto FOUND_MSB;
  } while (shift);

  /* if (data == 0) shift = DPH ? 16 ^ 31 : 0 ^ 31; */
  shift = 31 - 16 * cp2->doublePrecision;

FOUND_MSB:
  shift ^= 31;
  addr = (data << shift) >> 22;
  addr &= 0x000001FE;
  addr |= 0x00000200 | (shift & 1);
  fetch = ReciprocalLUT[addr];
  shift ^= 31;
  shift >>= 1;
  cp2->divOut = (0x40000000 | (fetch << 14)) >> shift;

  if (cp2->divIn < 0)
    cp2->divOut = ~cp2->divOut;
  else if (cp2->divIn == 0)
    cp2->divOut = 0x7FFFFFFF;
  else if (cp2->divIn == -32768)
    cp2->divOut = 0xFFFF0000;

#ifdef USE_SSE
  _mm_store_si128((__m128i*) accLow, vtShuf);
#else
#warning "Unimplemented function: RSPVRSQL (No SSE)."
#endif

  vd[delement & 0x7] = (short) cp2->divOut;
  cp2->doublePrecision = false;
}

/* ============================================================================
 *  Instruction: VSAR (Vector Accumulator Read (and Write))
 * ========================================================================= */
void
RSPVSAR(struct RSPCP2 *cp2, int16_t *vd, __m128i unused(vsReg),
  __m128i unused(vtReg), __m128i unused(vtShuf)) {
  int16_t *accLow = cp2->accumulatorLow.slices;
  int16_t *accMid = cp2->accumulatorMid.slices;
  int16_t *accHigh = cp2->accumulatorHigh.slices;
  unsigned element = cp2->iw >> 21 & 0xF;

  /* ==========================================================================
   * Even though `vt` is ignored in VSAR, according to official sources as well
   * as reversing, lots of games seem to specify it as nonzero, possibly to
   * avoid register stalling or other VU hazards.  Not really certain why yet.
   * ======================================================================= */

  element ^= 0x8;

  /* ==========================================================================
   * Or, for exception overrides, should this be `e &= 0x7;` ?
   * Currently this code is safer because &= is less likely to catch oddities.
   * Either way, documentation shows that the switch range is 0:2, not 8:A.
   * ======================================================================= */
  switch (element) {
    case 0:
      memcpy(vd, accHigh, sizeof(short) * 8);
      break;

    case 1:
      memcpy(vd, accMid, sizeof(short) * 8);
      break;

    case 2:
      memcpy(vd, accLow, sizeof(short) * 8);
      break;

    default:
      memset(vd, 0, sizeof(short) * 8);
  }
}

/* ============================================================================
 *  Instruction: VSUB (Vector Subtraction of Short Elements)
 * ========================================================================= */
void
RSPVSUB(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
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
  vtRegNeg = _mm_cmplt_epi16(vtShuf, _mm_setzero_si128());
  vtRegPos = _mm_cmpeq_epi16(vtRegNeg, _mm_setzero_si128());

  vdReg = _mm_subs_epi16(vsReg, vtShuf);
  vMask = _mm_cmpeq_epi16(unsatDiff, vdReg);
  vMask = _mm_and_si128(vtRegNeg, vMask);

  vtRegNeg = _mm_and_si128(vMask, carryOut);
  vtRegPos = _mm_and_si128(vtRegPos, carryOut);
  carryOut = _mm_or_si128(vtRegNeg, vtRegPos);
  vdReg = _mm_subs_epi16(vdReg, carryOut);

  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) accLow, vaccLow);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), _mm_setzero_si128());
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), _mm_setzero_si128());
#else
#warning "Unimplemented function: RSPVSUB (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VSUBC (Vector Subtraction of Short Elements with Carry)
 * ========================================================================= */
void
RSPVSUBC(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i satDiff, lessThanMask, notEqualMask, vdReg;

  satDiff = _mm_subs_epu16(vsReg, vtShuf);
  vdReg = _mm_sub_epi16(vsReg, vtShuf);

  /* Set the carry out flags when difference is < 0. */
  notEqualMask = _mm_cmpeq_epi16(vdReg, _mm_setzero_si128());
  notEqualMask = _mm_cmpeq_epi16(notEqualMask, _mm_setzero_si128());
  lessThanMask = _mm_cmpeq_epi16(satDiff, _mm_setzero_si128());
  lessThanMask = _mm_and_si128(lessThanMask, notEqualMask);

  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) accLow, vdReg);
  _mm_store_si128((__m128i*) (cp2->vcolo.slices), lessThanMask);
  _mm_store_si128((__m128i*) (cp2->vcohi.slices), notEqualMask);
#else
#warning "Unimplemented function: RSPVSUBC (No SSE)."
#endif
}

/* ============================================================================
 *  Instruction: VXOR (Vector XOR of Short Elements)
 * ========================================================================= */
void
RSPVXOR(struct RSPCP2 *cp2, int16_t *vd,
  __m128i vsReg, __m128i unused(vtReg), __m128i vtShuf) {
  int16_t *accLow = cp2->accumulatorLow.slices;

#ifdef USE_SSE
  __m128i vdReg;

  vdReg = _mm_xor_si128(vtShuf, vsReg);

  _mm_store_si128((__m128i*) vd, vdReg);
  _mm_store_si128((__m128i*) accLow, vdReg);
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

    RSPVectorFunctionTable[cp2->opcode.id](cp2, vd, vs, vt, vtShuf);
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

