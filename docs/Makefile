# Minimal makefile for the nilmframe documentation.
SPHINXBUILD ?= python3 -m sphinx
SOURCEDIR    = .
BUILDDIR     = _build

.PHONY: help html doctest linkcheck clean serve

help:
	@echo "html      build the HTML documentation"
	@echo "doctest   run every example in the docstrings"
	@echo "clean     remove the build and the generated API pages"
	@echo "serve     build, then serve on http://localhost:8000"

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

# Examples are not decorative: this is what stops them rotting.
#
# autosummary writes the per-object pages during the html build, and the doctest
# builder collects examples from those pages -- so running doctest on a clean
# tree finds nothing at all and reports success. Generate first.
doctest: api/generated
	$(SPHINXBUILD) -b doctest $(SOURCEDIR) $(BUILDDIR)/doctest

api/generated:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html -q

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck

clean:
	rm -rf $(BUILDDIR) api/generated

serve: html
	python3 -m http.server -d $(BUILDDIR)/html 8000
