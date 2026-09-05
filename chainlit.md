# Ask the corpus

This interface answers questions from the documents that have been ingested
into the local store — and from nothing else.

**How it works.** Your question is embedded, matched against every stored
passage by vector similarity, and the shortlist is rescored by a cross-encoder
when one is configured. The surviving passages are assembled into a versioned
prompt and sent to a local model. Nothing leaves this machine.

**Reading an answer.** Bracketed markers such as `[2]` refer to the sources
listed beside the answer; open one to see the passage, its page, and its
score. If no passage matches your question, you are told so and the model is
not asked — an answer with no sources would be a guess.

**Rating an answer.** The buttons under each answer record whether it was
useful, together with the passages it was given. Retrieval quality here is
genuinely unmeasured; those ratings are the beginning of measuring it.

Passages are quoted to the model as data, never as instructions. Text inside a
document that appears to address the model is reported, not obeyed.
