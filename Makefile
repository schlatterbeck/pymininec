# To use this Makefile, get a copy of my SF Release Tools
# git clone git://git.code.sf.net/p/sfreleasetools/code sfreleasetools
# And point the environment variable RELEASETOOLS to the checkout
ifeq (,${RELEASETOOLS})
    RELEASETOOLS=../releasetools
endif
LASTRELEASE:=$(shell $(RELEASETOOLS)/lastrelease -n -rv)
VERSIONPY=mininec/Version.py
VERSION=$(VERSIONPY)
README=README.rst
PROJECT=pymininec

MINI=$(wildcard test/*.mini)
CR=$(patsubst %.mini,%.CR,$(notdir $(MINI)))

all: $(VERSION)

%.CR: test/%.mini
	tr '\n' '\r' < $< > $@

coverage:
	$(PYTHON) -m pytest --cov-report term-missing --cov mininec test

test:
	$(PYTHON) -m pytest test

# Generate basic input files with carriage return instead of linefeed
basic_input: $(CR)

clean:
	rm -f $(CR) README.html MININEC.INP mininec/Version.py announce_pypi
	rm -rf dist build upload upload_homepage ReleaseNotes.txt zoppel.png \
            pymininec.egg-info

.PHONY: clean test coverage basic_input

include $(RELEASETOOLS)/Makefile-pyrelease
