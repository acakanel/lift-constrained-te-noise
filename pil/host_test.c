/* host_test.c -- runs the deployed law on the host for a like-for-like
 * double-precision comparison.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ctrl.h"

#define NREF 64

static double amax(double a, double b) { return a > b ? a : b; }

static int part1(void)
{
    FILE *f = fopen("reference_vectors.csv", "r");
    char line[2048];
    double eCL = 0, eG = 0, eS = 0, eGS = 0, eSig = 0;
    double sCL = 0, sG = 0, sS = 0, sSig = 0;
    int n = 0, j;

    if (!f) { fprintf(stderr, "reference_vectors.csv not found\n"); return 1; }
    if (!fgets(line, sizeof line, f)) { fclose(f); return 1; }

    while (fgets(line, sizeof line, f)) {
        double v[15];
        char *p = line;
        float x[4], cl, gcl[4], spl, gspl[4], sig;
        for (j = 0; j < 15; ++j) { v[j] = strtod(p, &p); if (*p == ',') ++p; }
        for (j = 0; j < 4; ++j) x[j] = (float)v[j];

        gp_cl(x, &cl, gcl);
        gp_spl(x, &spl, gspl);
        sig = gp_spl_sigma(x);

        eCL = amax(eCL, fabs(cl - v[4]));
        sCL = amax(sCL, fabs(v[4]));
        for (j = 0; j < 4; ++j) {
            eG = amax(eG, fabs(gcl[j] - v[5 + j]));
            sG = amax(sG, fabs(v[5 + j]));
            eGS = amax(eGS, fabs(gspl[j] - v[10 + j]));
        }
        eS = amax(eS, fabs(spl - v[9]));
        sS = amax(sS, fabs(v[9]));
        eSig = amax(eSig, fabs(sig - v[14]));
        sSig = amax(sSig, fabs(v[14]));
        ++n;
    }
    fclose(f);

    printf("static agreement over %d points (single vs double precision)\n", n);
    printf("  C_L            max |dev| %.3e   (range %.3f)\n", eCL, sCL);
    printf("  dC_L/dx        max |dev| %.3e   (range %.3f)\n", eG, sG);
    printf("  objective      max |dev| %.3e   (range %.3f)\n", eS, sS);
    printf("  d(objective)/dx max |dev| %.3e\n", eGS);
    printf("  predictive std max |dev| %.3e   (range %.3f)\n", eSig, sSig);
    return 0;
}

static void sched(int sc, double t, float *U, float *CL)
{
    if (sc == 0) { *U = 55.0f; *CL = 1.30f; return; }
    {   double r = t < 3.0 ? t / 3.0 : 1.0;
        *U  = (float)(70.0 - 25.0 * r);
        *CL = (float)(1.00 + 0.45 * r); }
}

static void part2(void)
{
    FILE *o = fopen("host_loop.csv", "w");
    const float u0[2][3] = { { 9.0f, 10.0f, -8.0f }, { 5.0f, 0.0f, 0.0f } };
    const double T[2] = { 20.0, 6.0 };
    const float preview[2] = { 0.4f, 0.8f };
    int sc, i, n;

    fprintf(o, "scenario,i,t,alpha,delta1,delta2,CL,s,sigma,flags\n");
    for (sc = 0; sc < 2; ++sc) {
        ctrl_state_t st;
        ctrl_out_t out;
        ctrl_init(&st, u0[sc]);
        n = (int)(T[sc] / CTRL_DT + 0.5);
        for (i = 0; i < n; ++i) {
            double t = i * (double)CTRL_DT;
            float U, CL, Up, CLp;
            sched(sc, t, &U, &CL);
            sched(sc, t + preview[sc], &Up, &CLp);
            ctrl_in_t in;
            in.U = U; in.CL_req = CL; in.CL_meas = 0.0f;
            in.d[0] = in.d[1] = in.d[2] = 0.0f;
            in.U_preview = Up; in.CL_req_preview = CLp;
            in.preview = preview[sc]; in.mode = 0u;
            ctrl_step(&st, &in, &out);
            fprintf(o, "%d,%d,%.4f,%.7e,%.7e,%.7e,%.7e,%.7e,%.7e,%u\n",
                    sc, i, t, st.u[0], st.u[1], st.u[2],
                    out.CL, out.s, out.sigma, out.flags);
        }
        printf("scenario %d: settled at alpha=%.3f d1=%.3f d2=%.3f, |s|=%.2e\n",
               sc, st.u[0], st.u[1], st.u[2], fabs(out.s));
    }
    fclose(o);
    printf("wrote host_loop.csv\n");
}

int main(void)
{
    if (part1()) return 1;
    printf("\n");
    part2();
    return 0;
}
