/* pil_target.c -- STM32 target entry point: serves the host link and runs
 * the control step from the DWT cycle counter.
 */
#include <string.h>
#include "main.h"
#include "pil_proto.h"
#include "pil_core.h"

#define DWT_LAR_ADDR 0xE0001FB0u

static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    *(volatile uint32_t *)DWT_LAR_ADDR = 0xC5ACCE55u;
    DWT->CYCCNT = 0u;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static uint32_t dwt_cyc(void) { return DWT->CYCCNT; }

static UART_HandleTypeDef *H;

static int rxn(uint8_t *p, uint16_t n, uint32_t to)
{
    HAL_StatusTypeDef st = HAL_UART_Receive(H, p, n, to);
    if (st == HAL_OK) return 1;

    __HAL_UART_CLEAR_OREFLAG(H);
    __HAL_UART_CLEAR_FEFLAG(H);
    __HAL_UART_CLEAR_NEFLAG(H);
    H->ErrorCode = HAL_UART_ERROR_NONE;
    H->RxState = HAL_UART_STATE_READY;
    return 0;
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
    HAL_UART_Transmit(H, f, (uint16_t)(len + 6), HAL_MAX_DELAY);
}

static uint8_t request(uint8_t *pay, uint8_t *len)
{
    uint8_t b, hdr[2], buf[PIL_MAXPAY + 4], n;

    for (;;) {
        if (!rxn(&b, 1, 500u)) return 0u;
        if (b != PIL_REQ_SYNC0) continue;
        if (!rxn(&b, 1, 50u)) return 0u;
        if (b == PIL_REQ_SYNC1) break;
    }
    if (!rxn(hdr, 2, 50u)) return 0u;
    n = hdr[1];
    if (n > PIL_MAXPAY) return 0u;
    if (!rxn(buf + 2, (uint16_t)(n + 2u), 50u)) return 0u;
    buf[0] = hdr[0]; buf[1] = n;
    if (pil_crc16(buf, (uint32_t)n + 2u)
        != (uint16_t)(buf[2 + n] | ((uint16_t)buf[3 + n] << 8))) {
        uint32_t e = 1u; reply(PIL_CMD_ERR, &e, 4); return 0u;
    }
    memcpy(pay, buf + 2, n);
    *len = n;
    return hdr[0];
}

void pil_server(UART_HandleTypeDef *huart)
{
    uint8_t pay[PIL_MAXPAY], rsp[PIL_MAXPAY], len, rlen, cmd, rc;
    uint32_t cache;

    H = huart;
    dwt_init();
    cache = ((SCB->CCR & SCB_CCR_IC_Msk) ? 1u : 0u)
          | ((SCB->CCR & SCB_CCR_DC_Msk) ? 2u : 0u);
    pil_core_init(dwt_cyc, SystemCoreClock, cache);

    for (;;) {
        cmd = request(pay, &len);
        if (!cmd) continue;
        rlen = 0u;
        rc = pil_core_handle(cmd, pay, len, rsp, &rlen);
        reply(rc, rsp, rlen);
    }
}
