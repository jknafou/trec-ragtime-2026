"""TREC RAGTIME 2026: an agentic multilingual-to-English cited-report RAG pipeline.

One pipeline serves all three tracks. A request is decomposed into nuggets, the
nuggets are answered by parallel RAG loops against a hybrid multilingual index, a
coverage audit grows the nugget bank until saturation, and a deterministic selection
step serializes the result for report generation, retrieval and auto-nuggetization.

The packages are layered, and each imports only from those to its left::

    common -> config -> serving -> orchestration -> {preprocess, retrieval} -> pipeline

``common`` holds the primitives (unicode normalization, IDs, record schemas, artifact
IO, the run-directory layout, the passage store) and depends on nothing else here. One
package sits off that line: ``devkit`` holds the long-lived retrieval service, which
imports the layers to its left and is imported by none of them.

The layering is a module-level property. ``orchestration.cli`` is the one dispatcher that
must reach forward into ``preprocess`` and ``pipeline`` to run a stage, and it does so with
the import deferred into the function that dispatches, so the module graph stays acyclic
and importing ``ragtime.orchestration`` never drags a stage in.
"""
