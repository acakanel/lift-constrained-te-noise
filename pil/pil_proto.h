/* pil_proto.h -- wire protocol shared by target and host. */
#ifndef PIL_PROTO_H
#define PIL_PROTO_H

#include <stdint.h>

#define PIL_REQ_SYNC0   0xA5u
#define PIL_REQ_SYNC1   0x5Au
#define PIL_RSP_SYNC0   0x5Au
#define PIL_RSP_SYNC1   0xA5u

#define PIL_CMD_PING    0x01u
#define PIL_CMD_INIT    0x02u
#define PIL_CMD_STEP    0x03u
#define PIL_CMD_BENCH   0x04u
#define PIL_CMD_FAULT   0x05u
#define PIL_CMD_ERR     0x7Fu

#define PIL_MAGIC       0x4C495000u
#define PIL_MAXPAY      128u

#define PIL_FAULT_NAN_CL      0u
#define PIL_FAULT_NAN_U       1u
#define PIL_FAULT_OOR_OP      2u
#define PIL_FAULT_STEP_CMD    3u
#define PIL_FAULT_STUCK_SURF  4u
#define PIL_FAULT_N           5u

static inline uint16_t pil_crc16(const uint8_t *p, uint32_t n)
{
    uint16_t c = 0xFFFFu;
    uint32_t i;
    int b;
    for (i = 0; i < n; ++i) {
        c ^= (uint16_t)p[i] << 8;
        for (b = 0; b < 8; ++b)
            c = (c & 0x8000u) ? (uint16_t)((c << 1) ^ 0x1021u) : (uint16_t)(c << 1);
    }
    return c;
}

#endif
