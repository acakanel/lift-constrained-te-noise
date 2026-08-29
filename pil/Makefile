# Makefile -- workstation builds and the target footprint measurement.
#
#   make            build the host verification and the target stand-in
#   make verify     static agreement + kinematic trajectory on the host
#   make sim        full driver run against the stand-in
#   make footprint  cross-compile for Cortex-M7 and report the deployed size
#   make model      regenerate gp_model.[ch] and the reference vectors

CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra
LDLIBS  ?= -lm

ARMCC   ?= arm-none-eabi-gcc
ARMSIZE ?= arm-none-eabi-size
ARMFLAGS = -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard \
           -O2 -ffunction-sections -fdata-sections -Wall -Wextra

SRC = ctrl.c gp_model.c
HDR = ctrl.h gp_model.h pil_proto.h pil_core.h

all: host_test sim_target

host_test: host_test.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ host_test.c $(SRC) $(LDLIBS)

sim_target: sim_target.c pil_core.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ sim_target.c pil_core.c $(SRC) $(LDLIBS)

model:
	python3 export_model.py
	python3 ref_loop.py

verify: host_test
	./host_test

sim: sim_target
	python3 pil_host.py --sim

footprint: $(SRC) pil_core.c $(HDR)
	@$(ARMCC) $(ARMFLAGS) -c ctrl.c     -o /tmp/ctrl.o
	@$(ARMCC) $(ARMFLAGS) -c gp_model.c -o /tmp/gp_model.o
	@$(ARMCC) $(ARMFLAGS) -c pil_core.c -o /tmp/pil_core.o
	@echo "Cortex-M7, hard float, -O2"
	@$(ARMSIZE) -t -d /tmp/ctrl.o /tmp/gp_model.o /tmp/pil_core.o
	@echo
	@$(ARMSIZE) -A -d /tmp/gp_model.o | grep rodata

clean:
	rm -f host_test sim_target /tmp/ctrl.o /tmp/gp_model.o /tmp/pil_core.o

.PHONY: all model verify sim footprint clean
