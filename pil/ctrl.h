/* ctrl.h -- single-precision constraint layer, confidence gate and set-point
 * map. No dynamic allocation; <math.h> only; builds for host and Cortex-M7.
 * Units: degrees, m/s, dimensionless lift, seconds.
 */
#ifndef CTRL_H
#define CTRL_H

#include <stdint.h>
#include "gp_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CTRL_DT          0.01f
#define CTRL_K1          10.0f
#define CTRL_K2          40.0f
#define CTRL_MU          0.4f
#define CTRL_GAMMA       3.0f
#define CTRL_CONF_C      3.0f
#define CTRL_SETPOINT_DT 0.2f
#define CTRL_SP_EVERY    20u
#define CTRL_A_RATE      10.0f
#define CTRL_D_RATE      60.0f

#define CTRL_F_POS_SAT   0x01u
#define CTRL_F_RATE_SAT  0x02u
#define CTRL_F_SP_CLAMP  0x04u
#define CTRL_F_LOW_CONF  0x08u
#define CTRL_F_NONFINITE 0x10u
#define CTRL_F_ILLCOND   0x20u

#define CTRL_USE_MEAS_CL 0x01u
#define CTRL_USE_MEAS_D  0x02u

typedef struct {
    float    u[3];
    float    w;
    float    ustar[3];
    float    gamma_eff;
    float    t;
    uint32_t k;
} ctrl_state_t;

typedef struct {
    float    U;
    float    CL_req;
    float    CL_meas;
    float    d[3];
    float    U_preview;
    float    CL_req_preview;
    float    preview;
    uint32_t mode;
} ctrl_in_t;

typedef struct {
    float    rate[3];
    float    u[3];
    float    CL;
    float    s;
    float    sigma;
    uint32_t flags;
} ctrl_out_t;

void  gp_cl(const float x[4], float *value, float grad[4]);
void  gp_spl(const float x[4], float *value, float grad[4]);

float gp_spl_sigma(const float x[4]);

void  sp_lookup(float U, float CL_req, float ustar[3], float *sigma,
                uint32_t *flags);

void  ctrl_init(ctrl_state_t *st, const float u0[3]);

void  ctrl_step(ctrl_state_t *st, const ctrl_in_t *in, ctrl_out_t *out);

#ifdef __cplusplus
}
#endif
#endif
