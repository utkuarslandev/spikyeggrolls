PROFILE ?= local

.PHONY: bootstrap-local bootstrap-runpod doctor smoke tune full doctor-local doctor-runpod smoke-local smoke-runpod tune-local tune-runpod full-local full-runpod

bootstrap-local:
	bash scripts/bootstrap_local.sh

bootstrap-runpod:
	bash scripts/bootstrap_runpod.sh

doctor:
	bash scripts/doctor.sh $(PROFILE)

smoke:
	PROFILE=$(PROFILE) bash scripts/run_smoke.sh

tune:
	PROFILE=$(PROFILE) bash scripts/run_tune.sh

full:
	PROFILE=$(PROFILE) bash scripts/run_full.sh

doctor-local:
	PROFILE=local bash scripts/doctor.sh

doctor-runpod:
	PROFILE=runpod bash scripts/doctor.sh

smoke-local:
	PROFILE=local bash scripts/run_smoke.sh

smoke-runpod:
	PROFILE=runpod bash scripts/run_smoke.sh

tune-local:
	PROFILE=local bash scripts/run_tune.sh

tune-runpod:
	PROFILE=runpod bash scripts/run_tune.sh

full-local:
	PROFILE=local bash scripts/run_full.sh

full-runpod:
	PROFILE=runpod bash scripts/run_full.sh
