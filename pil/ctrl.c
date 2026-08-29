/* ctrl.c -- implementation of the interface declared in ctrl.h. */
#include "ctrl.h"
#include <math.h>

static float clampf(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

static int finitef_(float x)
{
    return (x - x) == 0.0f;
}

static void gp_eval(const float x[GP_D], const float inv_ls2[GP_D],
                    const float *alpha, float ym, float ys,
                    float *value, float grad[GP_D], float *kv)
{
    float z[GP_D], acc[GP_D], sum = 0.0f;
    int i, j;

    for (j = 0; j < GP_D; ++j) {
        z[j] = (x[j] - gp_in_mu[j]) / gp_in_sd[j];
        acc[j] = 0.0f;
    }

    for (i = 0; i < GP_N; ++i) {
        const float *xi = &gp_xn[i * GP_D];
        float d[GP_D], q = 0.0f, k, ak;
        for (j = 0; j < GP_D; ++j) {
            d[j] = z[j] - xi[j];
            q += d[j] * d[j] * inv_ls2[j];
        }
        k = expf(-0.5f * q);
        if (kv) kv[i] = k;
        ak = alpha[i] * k;
        sum += ak;
        if (grad) {
            for (j = 0; j < GP_D; ++j)
                acc[j] -= ak * d[j] * inv_ls2[j];
        }
    }

    if (value) *value = sum * ys + ym;
    if (grad) {
        for (j = 0; j < GP_D; ++j)
            grad[j] = ys * acc[j] / gp_in_sd[j];
    }
}

void gp_cl(const float x[GP_D], float *value, float grad[GP_D])
{
    gp_eval(x, gp_cl_inv_ls2, gp_cl_alpha, gp_cl_ym, gp_cl_ys, value, grad, 0);
}

void gp_spl(const float x[GP_D], float *value, float grad[GP_D])
{
    gp_eval(x, gp_spl_inv_ls2, gp_spl_alpha, gp_spl_ym, gp_spl_ys, value, grad, 0);
}

float gp_spl_sigma(const float x[GP_D])
{
    static float kv[GP_N];
    float var = 1.0f;
    int i, j;

    gp_eval(x, gp_spl_inv_ls2, gp_spl_alpha, gp_spl_ym, gp_spl_ys, 0, 0, kv);

    for (i = 0; i < GP_N; ++i) {
        const float *Li = &gp_spl_chol[(i * (i + 1)) / 2];
        float s = kv[i];
        for (j = 0; j < i; ++j)
            s -= Li[j] * kv[j];
        kv[i] = s / Li[i];
    }
    for (i = 0; i < GP_N; ++i)
        var -= kv[i] * kv[i];
    if (var < 0.0f) var = 0.0f;
    return sqrtf(var) * gp_spl_ys;
}

static int locate(const float *g, int n, float v, float *t)
{
    int i;
    if (v <= g[0])     { *t = 0.0f; return 0; }
    if (v >= g[n - 1]) { *t = 1.0f; return n - 2; }
    for (i = 0; i < n - 2; ++i)
        if (v < g[i + 1]) break;
    *t = (v - g[i]) / (g[i + 1] - g[i]);
    return i;
}

void sp_lookup(float U, float CL_req, float ustar[3], float *sigma,
               uint32_t *flags)
{
    float tu, tc, w00, w01, w10, w11;
    int iu, ic, k;

    if (flags && (U < sp_u_grid[0] || U > sp_u_grid[SP_NU - 1] ||
                  CL_req < sp_cl_grid[0] || CL_req > sp_cl_grid[SP_NCL - 1]))
        *flags |= CTRL_F_SP_CLAMP;

    iu = locate(sp_u_grid, SP_NU, U, &tu);
    ic = locate(sp_cl_grid, SP_NCL, CL_req, &tc);

    iu = (tu > 0.5f) ? iu + 1 : iu;
    ic = (tc > 0.5f) ? ic + 1 : ic;
    if (iu > SP_NU - 1)  iu = SP_NU - 1;
    if (ic > SP_NCL - 1) ic = SP_NCL - 1;
    (void)w00; (void)w01; (void)w10; (void)w11;

#define SPT(a, b, c) sp_table[(((a) * SP_NCL) + (b)) * 3 + (c)]
    for (k = 0; k < 3; ++k)
        ustar[k] = SPT(iu, ic, k);
#undef SPT
#define SPG(a, b) sp_gate[((a) * SP_NCL) + (b)]
    if (sigma)
        *sigma = SPG(iu, ic);
#undef SPG

    ustar[0] = clampf(ustar[0], CTRL_A_LO, CTRL_A_HI);
    ustar[1] = clampf(ustar[1], -CTRL_D_LIM, CTRL_D_LIM);
    ustar[2] = clampf(ustar[2], -CTRL_D_LIM, CTRL_D_LIM);
}

static const float CTRL_LO[3]   = { CTRL_A_LO, -CTRL_D_LIM, -CTRL_D_LIM };
static const float CTRL_HI[3]   = { CTRL_A_HI,  CTRL_D_LIM,  CTRL_D_LIM };
static const float CTRL_RATE[3] = { CTRL_A_RATE, CTRL_D_RATE, CTRL_D_RATE };

void ctrl_init(ctrl_state_t *st, const float u0[3])
{
    int k;
    for (k = 0; k < 3; ++k) {
        st->u[k] = clampf(u0[k], CTRL_LO[k], CTRL_HI[k]);
        st->ustar[k] = st->u[k];
    }
    st->w = 0.0f;
    st->gamma_eff = CTRL_GAMMA;
    st->t = 0.0f;
    st->k = 0u;
}

void ctrl_step(ctrl_state_t *st, const ctrl_in_t *in, ctrl_out_t *out)
{
    float x[GP_D], J[3], grad[GP_D], d[3];
    float U, CL_req, CL, s, JJ, Jp[3], phi1, phi2, sw, ff, as, rs;
    float ud[3], Je, du;
    uint32_t flags = 0u;
    int k;

    U = in->U; CL_req = in->CL_req;
    if (!finitef_(U) || !finitef_(CL_req)) {
        flags |= CTRL_F_NONFINITE;
        U = clampf(finitef_(U) ? U : 55.0f, 20.0f, 120.0f);
        CL_req = finitef_(CL_req) ? CL_req : 0.0f;
    }

    for (k = 0; k < 3; ++k) {
        d[k] = (in->mode & CTRL_USE_MEAS_D) ? in->d[k] : st->u[k];
        if (!finitef_(d[k])) { d[k] = st->u[k]; flags |= CTRL_F_NONFINITE; }
        d[k] = clampf(d[k], CTRL_LO[k], CTRL_HI[k]);
    }

    if (st->k % CTRL_SP_EVERY == 0u) {
        float sigma_map = 0.0f, sigma;
        sp_lookup(in->U_preview, in->CL_req_preview, st->ustar, &sigma_map, &flags);
        x[0] = d[0]; x[1] = d[1]; x[2] = d[2]; x[3] = U;
        sigma = gp_spl_sigma(x);
        if (!finitef_(sigma) || sigma < 0.0f) { sigma = 1.0f; flags |= CTRL_F_NONFINITE; }
        st->gamma_eff = CTRL_GAMMA / (1.0f + CTRL_CONF_C * sigma);
        if (st->gamma_eff < 0.5f * CTRL_GAMMA) flags |= CTRL_F_LOW_CONF;
    }

    x[0] = d[0]; x[1] = d[1]; x[2] = d[2]; x[3] = U;
    gp_cl(x, &CL, grad);
    J[0] = grad[0]; J[1] = grad[1]; J[2] = grad[2];
    if (in->mode & CTRL_USE_MEAS_CL) {
        if (finitef_(in->CL_meas)) CL = in->CL_meas;
        else                       flags |= CTRL_F_NONFINITE;
    }
    s = CL - CL_req;

    JJ = J[0] * J[0] + J[1] * J[1] + J[2] * J[2] + 1.0e-9f;
    if (JJ < 1.0e-6f) flags |= CTRL_F_ILLCOND;
    for (k = 0; k < 3; ++k) Jp[k] = J[k] / JJ;

    as = fabsf(s);
    sw = (s > 0.0f) ? 1.0f : (s < 0.0f ? -1.0f : 0.0f);
    phi1 = (sqrtf(as) + CTRL_MU * as * sqrtf(as)) * sw;

    phi2 = (0.5f + 2.0f * CTRL_MU * as
                 + 1.5f * CTRL_MU * CTRL_MU * as * as) * sw;

    ff = (in->CL_req_preview - CL_req)
       / (in->preview > 1.0e-6f ? in->preview : 1.0e-6f);
    if (!finitef_(ff)) { ff = 0.0f; flags |= CTRL_F_NONFINITE; }
    rs = -CTRL_K1 * phi1 + st->w + ff;
    st->w += -CTRL_K2 * phi2 * CTRL_DT;

    Je = 0.0f;
    for (k = 0; k < 3; ++k) Je += J[k] * (d[k] - st->ustar[k]);
    for (k = 0; k < 3; ++k) {
        float e = (d[k] - st->ustar[k]) - Jp[k] * Je;
        ud[k] = Jp[k] * rs - st->gamma_eff * e;
    }

    for (k = 0; k < 3; ++k) {
        if (!finitef_(ud[k])) { ud[k] = 0.0f; flags |= CTRL_F_NONFINITE; }
        if (ud[k] >  CTRL_RATE[k]) { ud[k] =  CTRL_RATE[k]; flags |= CTRL_F_RATE_SAT; }
        if (ud[k] < -CTRL_RATE[k]) { ud[k] = -CTRL_RATE[k]; flags |= CTRL_F_RATE_SAT; }
        du = d[k] + ud[k] * CTRL_DT;
        if (du < CTRL_LO[k]) { du = CTRL_LO[k]; flags |= CTRL_F_POS_SAT; }
        if (du > CTRL_HI[k]) { du = CTRL_HI[k]; flags |= CTRL_F_POS_SAT; }
        st->u[k] = du;
    }

    st->k += 1u;
    st->t = (float)st->k * CTRL_DT;

    if (out) {
        for (k = 0; k < 3; ++k) { out->rate[k] = ud[k]; out->u[k] = st->u[k]; }
        out->CL = CL;
        out->s = s;
        out->sigma = (CTRL_GAMMA / st->gamma_eff - 1.0f) / CTRL_CONF_C;
        out->flags = flags;
    }
}
