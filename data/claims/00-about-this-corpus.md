<!-- retrievable: false, a note to the reader, not an answer to any caller question. -->
## About this corpus
**Fictional.** Every document in `data/claims/` describes PINEWRIGHT MUTUAL, an insurer
that does not exist. Policy numbers, deductibles, phone numbers, hours and claim
handling steps are invented for a personal prototype. They describe no real insurer,
no real policy and no real claim.

Nothing here is insurance advice, a coverage determination, or a statement of what any
real policy covers. The intake flow in `app/claims.py` collects fields and files them to
a mock endpoint. No insurer receives anything and no claim is opened.

These documents are deliberately separate from `data/rental/`. The rental corpus is the
one the voice assistant retrieves over; this one is not indexed and is not part of the
57 question retrieval set or the grounding set.
