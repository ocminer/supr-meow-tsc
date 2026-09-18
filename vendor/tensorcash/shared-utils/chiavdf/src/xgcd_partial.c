/*
    Copyright (C) 2012 William Hart

    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files (the "Software"), to deal in the Software
    without restriction, including without limitation the rights to use, copy, modify, merge,
    publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
    to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

    MIT licensing permission obtained January 13, 2020 by Chia Network Inc. from William Hart
    */

#ifndef _XGCD_PARTIAL
#define _XGCD_PARTIAL

#include <gmp.h>
#include <limits.h>

/*
 * LLP64-safe helpers for GMP functions that take long/unsigned long arguments.
 * On Windows 64-bit, long is 32-bit while mp_limb_signed_t is 64-bit, so
 * passing limb-sized values to mpz_mul_si / mpz_addmul_ui / mpz_submul_ui
 * truncates silently. These helpers use mpz_import for values that exceed
 * the native long/unsigned long range.
 *
 * The negative-value path uses ~x + 1u instead of -x to avoid undefined
 * behavior when x is the minimum signed value (two's complement).
 */
static inline mp_limb_t limb_abs(mp_limb_signed_t x)
{
    return x >= 0 ? (mp_limb_t)x : ~(mp_limb_t)x + (mp_limb_t)1u;
}
static inline void mpz_mul_limb_si(mpz_t rop, const mpz_t op, mp_limb_signed_t si)
{
    if (si >= LONG_MIN && si <= LONG_MAX) {
        mpz_mul_si(rop, op, (long)si);
        return;
    }
    mpz_t tmp;
    mpz_init(tmp);
    if (si >= 0) {
        mp_limb_t v = (mp_limb_t)si;
        mpz_import(tmp, 1, -1, sizeof(mp_limb_t), 0, 0, &v);
    } else {
        mp_limb_t v = ~(mp_limb_t)si + (mp_limb_t)1u;
        mpz_import(tmp, 1, -1, sizeof(mp_limb_t), 0, 0, &v);
        mpz_neg(tmp, tmp);
    }
    mpz_mul(rop, op, tmp);
    mpz_clear(tmp);
}

static inline void mpz_addmul_limb_ui(mpz_t rop, const mpz_t op, mp_limb_t ui)
{
    if (ui <= ULONG_MAX) {
        mpz_addmul_ui(rop, op, (unsigned long)ui);
        return;
    }
    mpz_t tmp;
    mpz_init(tmp);
    mpz_import(tmp, 1, -1, sizeof(mp_limb_t), 0, 0, &ui);
    mpz_addmul(rop, op, tmp);
    mpz_clear(tmp);
}

static inline void mpz_submul_limb_ui(mpz_t rop, const mpz_t op, mp_limb_t ui)
{
    if (ui <= ULONG_MAX) {
        mpz_submul_ui(rop, op, (unsigned long)ui);
        return;
    }
    mpz_t tmp;
    mpz_init(tmp);
    mpz_import(tmp, 1, -1, sizeof(mp_limb_t), 0, 0, &ui);
    mpz_submul(rop, op, tmp);
    mpz_clear(tmp);
}

void mpz_xgcd_partial(mpz_t co2, mpz_t co1,
                                    mpz_t r2, mpz_t r1, const mpz_t L)
{
   mpz_t q, r;
   mp_limb_signed_t aa2, aa1, bb2, bb1, rr1, rr2, qq, bb, t1, t2, t3, i;
   mp_limb_signed_t bits, bits1, bits2;

   mpz_init(q); mpz_init(r);

   mpz_set_ui(co2, 0);
   mpz_set_si(co1, -1);

   while (mpz_cmp_ui(r1, 0) && mpz_cmp(r1, L) > 0)
   {
      bits2 = mpz_sizeinbase(r2, 2);
      bits1 = mpz_sizeinbase(r1, 2);
      bits = __GMP_MAX(bits2, bits1) - GMP_LIMB_BITS + 1;
      if (bits < 0) bits = 0;

      /* Use mpz_getlimbn instead of mpz_get_ui to avoid truncation on
         Windows LLP64 where unsigned long is 32-bit but GMP limbs are 64-bit. */
      mpz_tdiv_q_2exp(r, r2, bits);
      rr2 = mpz_size(r) ? (mp_limb_signed_t)mpz_getlimbn(r, 0) : 0;
      mpz_tdiv_q_2exp(r, r1, bits);
      rr1 = mpz_size(r) ? (mp_limb_signed_t)mpz_getlimbn(r, 0) : 0;
      mpz_tdiv_q_2exp(r, L, bits);
      bb = mpz_size(r) ? (mp_limb_signed_t)mpz_getlimbn(r, 0) : 0;

      aa2 = 0; aa1 = 1;
      bb2 = 1; bb1 = 0;

      for (i = 0; rr1 != 0 && rr1 > bb; i++)
      {
         qq = rr2 / rr1;

         t1 = rr2 - qq*rr1;
         t2 = aa2 - qq*aa1;
         t3 = bb2 - qq*bb1;

         if (i & 1)
         {
            if (t1 < -t3 || rr1 - t1 < t2 - aa1) break;
         } else
         {
            if (t1 < -t2 || rr1 - t1 < t3 - bb1) break;
         }

         rr2 = rr1; rr1 = t1;
         aa2 = aa1; aa1 = t2;
         bb2 = bb1; bb1 = t3;
      }

      if (i == 0)
      {
         mpz_fdiv_qr(q, r2, r2, r1);
         mpz_swap(r2, r1);

         mpz_submul(co2, co1, q);
         mpz_swap(co2, co1);
      } else
      {
         mpz_mul_limb_si(r, r2, bb2);
         if (aa2 >= 0)
            mpz_addmul_limb_ui(r, r1, (mp_limb_t)aa2);
         else
            mpz_submul_limb_ui(r, r1, limb_abs(aa2));
         mpz_mul_limb_si(r1, r1, aa1);
         if (bb1 >= 0)
            mpz_addmul_limb_ui(r1, r2, (mp_limb_t)bb1);
         else
            mpz_submul_limb_ui(r1, r2, limb_abs(bb1));
         mpz_set(r2, r);

         mpz_mul_limb_si(r, co2, bb2);
         if (aa2 >= 0)
            mpz_addmul_limb_ui(r, co1, (mp_limb_t)aa2);
         else
            mpz_submul_limb_ui(r, co1, limb_abs(aa2));
         mpz_mul_limb_si(co1, co1, aa1);
         if (bb1 >= 0)
            mpz_addmul_limb_ui(co1, co2, (mp_limb_t)bb1);
         else
            mpz_submul_limb_ui(co1, co2, limb_abs(bb1));
         mpz_set(co2, r);

         if (mpz_sgn(r1) < 0) { mpz_neg(co1, co1); mpz_neg(r1, r1); }
         if (mpz_sgn(r2) < 0) { mpz_neg(co2, co2); mpz_neg(r2, r2); }
      }
   }

   if (mpz_sgn(r2) < 0)
   {
      mpz_neg(co2, co2); mpz_neg(co1, co1);
      mpz_neg(r2, r2);
   }

   mpz_clear(q); mpz_clear(r);
}
#endif /* _XGCD_PARTIAL */
