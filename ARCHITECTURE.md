# Architecture — functional core, imperative shell

ReDevOps runtimes use a **functional-core / imperative-shell** architecture. Canonical artifacts and
deterministic transformations remain value-oriented and side-effect free; runtime actors, external
capabilities, stores, and providers are exposed through explicit interfaces.

## Three layers

**Value contracts** (data) — `@dataclass(frozen=True)`: intents, plans, evidence, events, outcomes,
capability declarations, policy references. Easy to serialize, hash, compare, persist, replay, and
share across languages. No runtime logic lives inside them.

**Functional core** (pure functions) — canonicalization, compilation, hashing / fingerprinting,
reconciliation, scoring, fusion, coverage, verification predicates, replay reducers, event
reductions, financial operators. Same input → same output; no hidden state. This is what keeps the
system deterministic, replayable, formally verifiable, and mutation-testable.

**Imperative shell** (objects at the boundaries) — runtime actors (`DiscoveryRuntime`,
`MissionRuntime`, `ContextRuntime`), readers, capabilities, stores, providers, clients, and
registries. These own long-lived state, wrap external services, and are exposed as small `Protocol`
interfaces so implementations are interchangeable (`MemoryEventStore` / `PostgresEventStore`,
static / db-backed `PolicyProvider`, hosted / recorded readers, …).

```
imperative shell   → runtimes · readers · capabilities · stores · providers
functional core    → canonicalization · compilation · scoring · reconciliation · replay · verification
value contracts    → intents · plans · evidence · events · outcomes
```

## Design rules

- **Classes** when a component owns long-lived state, wraps an external service, has interchangeable
  implementations, has a lifecycle, or callers should depend on an interface rather than a concrete type.
- **Values** when the thing is evidence or a contract, must serialize, or where identity / hash matters
  or it crosses a runtime / language boundary.
- **Functions** when the same input must produce the same output, replay depends on the behavior, or it
  must be formally verifiable / easy to mutation-test.

We do **not** optimize for class count, inheritance depth, "everything is an object", or design-pattern
count. We optimize for explicit authority boundaries, determinism, replayability, independently testable
components, small public interfaces, low coupling, and replaceable external dependencies. **Runtime
actors depend on interfaces and canonical contracts, never on concrete adapters.**

The runtimes are **more object-oriented at the boundaries, and strictly functional in the deterministic
core** — this preserves the properties that matter most to ReDevOps: determinism, replay, portability,
evidence, and formal verification.
