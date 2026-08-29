/* sim_target.c -- target firmware compiled for the host, so the driver can
 * be exercised without a board attached.
 */
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "pil_proto.h"
#include "pil_core.h"

static uint32_t zero_cyc(void) { return 0u; }

static int rd(uint8_t *p, uint32_t n)
{
    uint32_t got = 0;
    while (got < n) {
        ssize_t k = read(0, p + got, n - got);
        if (k <= 0) return 0;
        got += (uint32_t)k;
    }
    return 1;
}

static void wr(const uint8_t *p, uint32_t n)
{
    uint32_t put = 0;
    while (put < n) {
        ssize_t k = write(1, p + put, n - put);
        if (k <= 0) return;
        put += (uint32_t)k;
    }
}

static void reply(uint8_t cmd, const void *pay, uint8_t len)
{
    uint8_t f[6 + PIL_MAXPAY];
    uint16_t c;
    f[0] = PIL_RSP_SYNC0; f[1] = PIL_RSP_SYNC1; f[2] = cmd; f[3] = len;
    if (len) memcpy(f + 4, pay, len);
    c = pil_crc16(f + 2, (uint32_t)len + 2u);
    f[4 + len] = (uint8_t)(c & 0xFFu);
    f[5 + len] = (uint8_t)(c >> 8);
    wr(f, (uint32_t)len + 6u);
}

int main(void)
{
    uint8_t pay[PIL_MAXPAY], rsp[PIL_MAXPAY], hdr[2], b, len, rlen, rc;

    pil_core_init(zero_cyc, 0u, 0u);

    for (;;) {
        do { if (!rd(&b, 1)) return 0; } while (b != PIL_REQ_SYNC0);
        if (!rd(&b, 1)) return 0;
        if (b != PIL_REQ_SYNC1) continue;
        if (!rd(hdr, 2)) return 0;
        len = hdr[1];
        if (len > PIL_MAXPAY) continue;
        if (len && !rd(pay, len)) return 0;
        {   uint8_t crc[2], buf[PIL_MAXPAY + 2];
            if (!rd(crc, 2)) return 0;
            buf[0] = hdr[0]; buf[1] = len;
            if (len) memcpy(buf + 2, pay, len);
            if (pil_crc16(buf, (uint32_t)len + 2u)
                != (uint16_t)(crc[0] | ((uint16_t)crc[1] << 8))) {
                uint32_t e = 1u; reply(PIL_CMD_ERR, &e, 4); continue; } }
        rlen = 0u;
        rc = pil_core_handle(hdr[0], pay, len, rsp, &rlen);
        reply(rc, rsp, rlen);
    }
}
