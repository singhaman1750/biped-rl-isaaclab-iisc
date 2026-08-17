# Plans Index

This directory holds the prescriptive documents of the simulation repository, the designs and implementation plans that propose changes to the code within it. A plan document states a problem, grounds it in evidence, surveys whatever literature bears on it, and then sets out the concrete edits that realise the remedy, ordinarily at the granularity of a named file and a named symbol. Documents that record established facts rather than proposed changes live in [../context](../context/README.md) instead.

## Document register

This directory is presently empty. No plan is specific to this repository alone.

Every implementation plan of the project so far has concerned the workspace as a whole, since the pipelines they modify span this repository, the vendored `rsl_rl`, and the Isaac Lab installation together. Those plans therefore live at the workspace level in `../../plans/`, indexed with their implementation status by [../../plans/README.md](../../plans/README.md). Several of them prescribe edits to files within this repository, notably the co-optimisation investigation plan, the learned model design, the symmetry augmentation brief, and the two gait documents, so a reader looking for the rationale behind a configuration or a reward term in this repository should begin at that index rather than here.

## What belongs here

A plan belongs in this directory when its every edit falls inside this repository and it requires no coordinated change to the workspace, the vendored libraries, or the container tooling. A plan for a new robot asset and its task registration, a plan to restructure the environment configuration layering, or a plan to add a reward family would all qualify. A plan that touches the training entry point together with the co-optimisation package and the `djinn` command surface would not, and belongs at the workspace level.

Where the distinction is genuinely unclear, prefer the workspace level, since a plan is more useful over-scoped and findable than correctly scoped and overlooked.

## Conventions

A plan names its target as a file path and a symbol, and quotes surrounding code to locate an edit. Those quotations drift as the sources change, so an edit should be located by symbol name and the quoted code treated as corroboration rather than as an address.

Each plan carries a status banner immediately beneath its title, stating whether its proposals are implemented, partially implemented, superseded, or outstanding, and the date that status was last verified against the live sources. The register above must record the same status, so that a reader may triage without opening the document. On completing a plan, update its banner and record the outcome, together with any divergence from the plan, in the corresponding context document, so that the plan remains a statement of intent and the context record remains the statement of fact.

Cross references within this directory use the bare filename. References to this repository's factual record take the form `../context/NAME.md`, to the workspace record `../../context/NAME.md`, and to the workspace plans `../../plans/NAME.md`.

The writing and change conventions governing these documents are those of the workspace, recorded in `../../CLAUDE.md`, which requires above all that backwards compatibility be preserved absolutely and that new behaviour be carried by an optional argument whose default reproduces the old behaviour exactly.
