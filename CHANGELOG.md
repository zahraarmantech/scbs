# Changelog

All notable changes to SCBS are documented in this file.

## [1.0.0] — Initial public release

### Added
- 10-slot blueprint extraction (WHO, ACTION, TECH, EMOTION, WHEN, SOCIAL, INTENT, MODIFIER, WORLD, DOMAIN)
- 930-word vocabulary with 50 fine-grained sub-clusters
- Domain-specific sub-clusters: Security Events, Incident Management, Financial Operations, HR Operations, Customer Operations
- Hybrid signature + zone index for sub-millisecond search
- TF-IDF weighted distance formula
- Co-occurrence clustering for corpus-driven filtering
- Per-session puzzle-piece exclusion for diverse results
- 4-layer unknown word resolver (exact, prefix/suffix, edit-distance, n-gram voting)
- Unified `Encoder` and `Store` public API
- 18 unit and integration tests
- Architecture documentation
- Working quickstart example

### Performance
- Search latency: 0.5ms at 1K records (16× faster than LLM Pinecone reference)
- Memory: 100× smaller than 768-dimensional LLM embeddings
- Write throughput: 1,000+ records/second on a single CPU core
- Zero external dependencies
- Zero monthly cost at any scale
