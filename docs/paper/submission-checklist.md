# What a journal submission still needs

Honest assessment as of this commit. Items are ordered by how likely a reviewer is to reject without them.

1. **Real or second public data.** The only data is a fictitious, single-design Iranian ID dataset. Add at
   least one of: (a) real cards with documented consent and ethics approval, (b) a public benchmark
   (e.g. MIDV-500/MIDV-2019 fields) to show the method is not specific to one synthetic design.
2. **Stronger baselines.** Add a modern open recogniser run under the same protocol (e.g. PaddleOCR or
   TrOCR fine-tuned on the same 1,376 photographs) and a CRNN trained with *manually aligned* crops
   (upper bound for pseudo-alignment).
3. **Repeated runs.** Several seeds of every stage (bootstrap and joint), report mean ± std; k-fold instead
   of one split.
4. **Calibration and operating point.** Choose τ on a validation split, report precision/coverage curves,
   expected calibration error; compare with checksum-based verification on valid-checksum data.
5. **Related work.** A real literature search (document localisation, weakly supervised OCR, KYC pipelines);
   every claim of novelty must be checked against it. CRNN/CTC for ID digits is *not* novel; the
   claimed novelty is the label-verified pseudo-alignment and joint registration-by-recognition.
6. **Venue.** Document-analysis and pattern-recognition journals (e.g. IJDAR, Pattern Recognition
   Letters, Expert Systems with Applications, Applied Soft Computing) are plausible; check current
   indexing, scope and review times at submission. No outcome can be promised.
7. **Reproducibility.** Code is public; weights cannot be released. Provide a seed-controlled script and
   a Docker/CI recipe; consider a synthetic-only release of weights.
