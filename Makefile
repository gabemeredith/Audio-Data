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
FILE ?=
BOOST ?= 0

export PYTHONUNBUFFERED := 1
RUN_CMD := uv run python -m audio_probes
OVERRIDES := $(filter-out PROPERTY=% CONDITION=% CLIP=% RUN=% FILE=% BOOST=%,$(MAKEOVERRIDES))
COND_FLAG := $(if $(CONDITION),--condition $(CONDITION),)

.PHONY: all data embed probe llm report inspect predict play demo test config download-librispeech

all: data embed probe llm report

data embed probe llm report:
	$(RUN_CMD) $@ --property $(PROPERTY) $(COND_FLAG) $(OVERRIDES)

inspect:
	$(RUN_CMD) inspect --property $(PROPERTY) $(COND_FLAG) --clip $(CLIP) $(if $(RUN),--run $(RUN),) $(OVERRIDES)

# Mac helpers (no microphone needed):
#   make predict PROPERTY=pitch FILE=/System/Library/Sounds/Glass.aiff
#   make play PROPERTY=distance CONDITION=raw CLIP=distance_00042 BOOST=20
predict:
	$(RUN_CMD) predict --property $(PROPERTY) $(COND_FLAG) --file "$(FILE)" $(OVERRIDES)

play:
	$(RUN_CMD) play --property $(PROPERTY) $(COND_FLAG) --clip $(CLIP) --boost-db $(BOOST) $(OVERRIDES)

# Interactive lab on http://127.0.0.1:8770 (needs `make all` first; ~1.5 GB while running)
demo:
	$(RUN_CMD).demo

config:
	$(RUN_CMD) show-config --property $(PROPERTY) $(OVERRIDES)

test:
	uv run pytest -q

# LibriSpeech dev-clean (~337 MB) supplies the dry speech used by the distance property.
download-librispeech:
	mkdir -p data/librispeech
	curl -L --fail -o data/librispeech/dev-clean.tar.gz https://www.openslr.org/resources/12/dev-clean.tar.gz
	tar -xzf data/librispeech/dev-clean.tar.gz -C data/librispeech
