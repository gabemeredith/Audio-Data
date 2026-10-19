# Pipeline stages. Each one reads the previous stage's cache from data/ and skips finished work.
#
#   make all                                   # every stage, every property
#   make probe PROPERTY=distance probe.alpha=10
#   make embed PROPERTY=pitch encoder.size=tiny
#   make inspect PROPERTY=distance CONDITION=raw CLIP=distance_00042
#
# Any `key.sub=value` on the make command line is forwarded as a config override.

PROPERTY ?= all
CONDITION ?=
CLIP ?=
RUN ?=

RUN_CMD := uv run python -m audio_probes
OVERRIDES := $(filter-out PROPERTY=% CONDITION=% CLIP=% RUN=%,$(MAKEOVERRIDES))
COND_FLAG := $(if $(CONDITION),--condition $(CONDITION),)

.PHONY: all data embed probe llm report inspect test config download-librispeech

all: data embed probe llm report

data embed probe llm report:
	$(RUN_CMD) $@ --property $(PROPERTY) $(COND_FLAG) $(OVERRIDES)

inspect:
	$(RUN_CMD) inspect --property $(PROPERTY) $(COND_FLAG) --clip $(CLIP) $(if $(RUN),--run $(RUN),) $(OVERRIDES)

config:
	$(RUN_CMD) show-config --property $(PROPERTY) $(OVERRIDES)

test:
	uv run pytest -q

# LibriSpeech dev-clean (~337 MB) supplies the dry speech used by the distance property.
download-librispeech:
	mkdir -p data/librispeech
	curl -L --fail -o data/librispeech/dev-clean.tar.gz https://www.openslr.org/resources/12/dev-clean.tar.gz
	tar -xzf data/librispeech/dev-clean.tar.gz -C data/librispeech
