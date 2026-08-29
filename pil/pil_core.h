/* pil_core.h -- framing and command interface for the host link. */
#ifndef PIL_CORE_H
#define PIL_CORE_H

#include <stdint.h>

typedef uint32_t (*pil_cyc_fn)(void);

void    pil_core_init(pil_cyc_fn cyc, uint32_t clk, uint32_t cache);

uint8_t pil_core_handle(uint8_t cmd, const uint8_t *pay, uint8_t len,
                        uint8_t *rsp, uint8_t *rlen);

#endif
