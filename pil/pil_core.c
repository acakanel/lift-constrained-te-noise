/* pil_core.c -- frame parsing, CRC-16/CCITT-FALSE and command dispatch. */
#include <string.h>
#include <math.h>
#include "ctrl.h"
#include "pil_proto.h"
#include "pil_core.h"

static pil_cyc_fn CYC;
static uint32_t   CLK, CACHE;

static ctrl_state_t ST;
static ctrl_in_t    IN;
static ctrl_out_t   OUT;

static uint32_t cyc(void) { return CYC ? CYC() : 0u; }

static uint32_t build_flags(void)
{
    uint32_t f = 0u;
#ifdef __OPTIMIZE__
    f |= 1u;
#endif
#ifdef __OPTIMIZE_SIZE__
    f |= 2u;
#endif
#ifdef __GNUC__
    f |= ((uint32_t)__GNUC__ & 0xFFu) << 8;
#endif
    return f;
}

void pil_core_init(pil_cyc_fn c, uint32_t clk, uint32_t cache)
{
    const float u0[3] = { 9.0f, 10.0f, -8.0f };
    CYC = c; CLK = clk; CACHE = cache;
    ctrl_init(&ST, u0);
}

typedef struct { uint32_t n, lo, hi; uint64_t sum; } stat_t;

static void st_reset(stat_t *s) { s->n = 0u; s->lo = 0xFFFFFFFFu; s->hi = 0u; s->sum = 0u; }
static void st_add(stat_t *s, uint32_t v)
{
    if (v < s->lo) s->lo = v;
    if (v > s->hi) s->hi = v;
    s->sum += v; s->n++;
}
static uint32_t st_mean(const stat_t *s) { return s->n ? (uint32_t)(s->sum / s->n) : 0u; }

uint8_t pil_core_handle(uint8_t cmd, const uint8_t *pay, uint8_t len,
                        uint8_t *rsp, uint8_t *rlen)
{
    if (cmd == PIL_CMD_PING) {
        uint32_t r[9];
        r[0] = PIL_MAGIC;
        r[1] = (uint32_t)GP_N;
        r[2] = (uint32_t)SP_NU;
        r[3] = (uint32_t)SP_NCL;
        r[4] = CLK;
        r[5] = CACHE & 1u;
        r[6] = (CACHE >> 1) & 1u;
        r[7] = (uint32_t)sizeof(float);
        r[8] = build_flags();
        memcpy(rsp, r, sizeof r); *rlen = (uint8_t)sizeof r;
        return PIL_CMD_PING;
    }

    if (cmd == PIL_CMD_INIT) {
        float u0[3] = { 9.0f, 10.0f, -8.0f };
        if (len >= 12u) memcpy(u0, pay, 12);
        ctrl_init(&ST, u0);
        memcpy(rsp, ST.u, 12); *rlen = 12u;
        return PIL_CMD_INIT;
    }

    if (cmd == PIL_CMD_STEP) {
        float f[10];
        uint32_t mode, t0, t1, c_outer = 0u, c_gp;
        if (len < 44u) { uint32_t e = 2u; memcpy(rsp, &e, 4); *rlen = 4u; return PIL_CMD_ERR; }
        memcpy(f, pay, 40);
        memcpy(&mode, pay + 40, 4);
        IN.U = f[0]; IN.CL_req = f[1]; IN.CL_meas = f[2];
        IN.d[0] = f[3]; IN.d[1] = f[4]; IN.d[2] = f[5];
        IN.U_preview = f[6]; IN.CL_req_preview = f[7]; IN.preview = f[8];
        IN.mode = mode;

        {   float x[4], v, g[4];
            x[0] = (mode & CTRL_USE_MEAS_D) ? IN.d[0] : ST.u[0];
            x[1] = (mode & CTRL_USE_MEAS_D) ? IN.d[1] : ST.u[1];
            x[2] = (mode & CTRL_USE_MEAS_D) ? IN.d[2] : ST.u[2];
            x[3] = IN.U;
            t0 = cyc(); gp_cl(x, &v, g); c_gp = cyc() - t0; (void)v; (void)g; }

        t0 = cyc();
        ctrl_step(&ST, &IN, &OUT);
        t1 = cyc() - t0;
        if (((ST.k - 1u) % CTRL_SP_EVERY) == 0u) c_outer = t1;

        memcpy(rsp +  0, OUT.rate, 12);
        memcpy(rsp + 12, OUT.u,    12);
        memcpy(rsp + 24, &OUT.CL,    4);
        memcpy(rsp + 28, &OUT.s,     4);
        memcpy(rsp + 32, &OUT.sigma, 4);
        memcpy(rsp + 36, &OUT.flags, 4);
        memcpy(rsp + 40, &t1,        4);
        memcpy(rsp + 44, &c_gp,      4);
        memcpy(rsp + 48, &c_outer,   4);
        *rlen = 52u;
        return PIL_CMD_STEP;
    }

    if (cmd == PIL_CMD_BENCH) {
        uint32_t n = 200u, i, t0, r[16];
        stat_t s_inner, s_outer, s_gp, s_sig, s_map;
        float x[4], v, g[4], us[3], sg = 0.0f;
        ctrl_state_t b;
        const float u0[3] = { 9.0f, 10.0f, -8.0f };

        if (len >= 4u) memcpy(&n, pay, 4);
        if (n == 0u || n > 5000u) n = 200u;
        st_reset(&s_inner); st_reset(&s_outer);
        st_reset(&s_gp); st_reset(&s_sig); st_reset(&s_map);

        for (i = 0; i < n; ++i) {
            x[0] = 4.0f + 6.0f * (float)(i % 7) / 6.0f;
            x[1] = -8.0f + 16.0f * (float)(i % 5) / 4.0f;
            x[2] = -8.0f + 16.0f * (float)(i % 3) / 2.0f;
            x[3] = 45.0f + 20.0f * (float)(i % 11) / 10.0f;
            t0 = cyc(); gp_cl(x, &v, g);                   st_add(&s_gp,  cyc() - t0);
            t0 = cyc(); sg = gp_spl_sigma(x);              st_add(&s_sig, cyc() - t0);
            t0 = cyc(); sp_lookup(x[3], 1.3f, us, &sg, 0); st_add(&s_map, cyc() - t0);
        }
        (void)v; (void)sg;

        ctrl_init(&b, u0);
        IN.U = 55.0f; IN.CL_req = 1.30f; IN.CL_meas = 0.0f;
        IN.d[0] = IN.d[1] = IN.d[2] = 0.0f;
        IN.U_preview = 55.0f; IN.CL_req_preview = 1.30f; IN.preview = 0.4f;
        IN.mode = 0u;
        for (i = 0; i < n; ++i) {
            t0 = cyc();
            ctrl_step(&b, &IN, &OUT);
            t0 = cyc() - t0;
            if ((i % CTRL_SP_EVERY) == 0u) st_add(&s_outer, t0);
            else                           st_add(&s_inner, t0);
        }

        r[0]  = n;
        r[1]  = CLK;
        r[2]  = s_inner.lo; r[3]  = st_mean(&s_inner); r[4]  = s_inner.hi;
        r[5]  = s_outer.lo; r[6]  = st_mean(&s_outer); r[7]  = s_outer.hi;
        r[8]  = s_gp.lo;    r[9]  = st_mean(&s_gp);    r[10] = s_gp.hi;
        r[11] = s_sig.lo;   r[12] = st_mean(&s_sig);   r[13] = s_sig.hi;
        r[14] = st_mean(&s_map);
        r[15] = CACHE;
        memcpy(rsp, r, sizeof r); *rlen = (uint8_t)sizeof r;
        return PIL_CMD_BENCH;
    }

    if (cmd == PIL_CMD_FAULT) {

        uint32_t kind = PIL_FAULT_N, nstep = 100u, i, acc = 0u, ok = 1u;
        const float u0[3] = { 2.0f, 10.0f, 10.0f };
        float smax = 0.0f, a;
        ctrl_state_t f;

        if (len >= 4u) memcpy(&kind, pay, 4);
        if (len >= 8u) memcpy(&nstep, pay + 4, 4);
        if (nstep == 0u || nstep > 2000u) nstep = 100u;

        ctrl_init(&f, u0);
        IN.U = 55.0f; IN.CL_req = 1.30f; IN.CL_meas = 1.30f;
        IN.U_preview = 55.0f; IN.CL_req_preview = 1.30f; IN.preview = 0.4f;
        IN.mode = 0u;
        switch (kind) {
        case PIL_FAULT_NAN_CL:   IN.CL_meas = (float)NAN;
                                 IN.mode = CTRL_USE_MEAS_CL; break;
        case PIL_FAULT_NAN_U:    IN.U = (float)NAN; IN.U_preview = (float)NAN; break;
        case PIL_FAULT_OOR_OP:   IN.U = 95.0f; IN.U_preview = 95.0f;
                                 IN.CL_req = 2.4f; IN.CL_req_preview = 2.4f; break;
        case PIL_FAULT_STEP_CMD: IN.CL_req = 0.60f; IN.CL_req_preview = 0.60f; break;
        default: break;
        }

        for (i = 0; i < nstep; ++i) {
            if (kind == PIL_FAULT_STUCK_SURF) {

                IN.mode = CTRL_USE_MEAS_D;
                IN.d[0] = f.u[0]; IN.d[1] = CTRL_D_LIM; IN.d[2] = f.u[2];
            }
            ctrl_step(&f, &IN, &OUT);
            acc |= OUT.flags;
            a = OUT.s < 0.0f ? -OUT.s : OUT.s;
            if (i > nstep / 2u && a > smax) smax = a;
            if (!(OUT.u[0] == OUT.u[0]) || !(OUT.u[1] == OUT.u[1])
                || !(OUT.u[2] == OUT.u[2])) ok = 0u;
            if (OUT.u[0] < CTRL_A_LO - 1e-3f || OUT.u[0] > CTRL_A_HI + 1e-3f) ok = 0u;
            if (OUT.u[1] < -CTRL_D_LIM - 1e-3f || OUT.u[1] > CTRL_D_LIM + 1e-3f) ok = 0u;
            if (OUT.u[2] < -CTRL_D_LIM - 1e-3f || OUT.u[2] > CTRL_D_LIM + 1e-3f) ok = 0u;
        }

        memcpy(rsp +  0, OUT.rate, 12);
        memcpy(rsp + 12, OUT.u,    12);
        memcpy(rsp + 24, &smax,     4);
        memcpy(rsp + 28, &acc,      4);
        memcpy(rsp + 32, &kind,     4);
        memcpy(rsp + 36, &ok,       4);
        memcpy(rsp + 40, &nstep,    4);
        *rlen = 44u;
        return PIL_CMD_FAULT;
    }

    {   uint32_t e = 3u; memcpy(rsp, &e, 4); *rlen = 4u; }
    return PIL_CMD_ERR;
}
