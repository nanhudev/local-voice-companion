# Architecture Decision Records

One file per significant decision. Each records the context, the decision, and
what it costs — an ADR that lists only benefits is not an ADR, it is a
justification.

| ADR | Decision |
| --- | --- |
| [0001](0001-provider-descriptor-as-contract.md) | One `ProviderDescriptor` as the anti-branch contract |
| [0002](0002-voicebox-and-ollama-as-providers.md) | Demote Voicebox and Ollama to providers |
| [0003](0003-selection-pipeline.md) | Candidate → Constraint → Score → Decision |
| [0004](0004-turn-state-and-barge-in.md) | Explicit turn state machine with token-based cancellation |
| [0005](0005-time-to-first-audio.md) | Time To First Audio as the headline metric |
| [0006](0006-compatibility-shim-over-rewrite.md) | Shim `app.py`, do not rewrite the client surface |
| [0007](0007-native-local-provider-tier.md) | A native local tier: faster-whisper CPU INT8 + Kokoro CPU |
| [0008](0008-g2p-as-a-first-class-dependency.md) | Grapheme-to-phoneme is its own dependency, probed and benchmarked |
| [0009](0009-piper-excluded-over-licensing.md) | Keep GPL-3.0 Piper out of the core distribution |
