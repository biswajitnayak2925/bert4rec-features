# bert4rec-features


This project investigates how the availability of heterogeneous item-side information affects the performance of BERT4Rec for sequential next-item prediction.
The central research question is:
How does the availability of heterogeneous item-side information affect the performance of BERT4Rec for sequential next-item prediction, and can adaptive gated fusion effectively exploit different levels of available information?
The project extends BERT4Rec by incorporating movie-level side information such as:

Genres
Keywords
Movie overview/text
Cast
Crew/director information

The main focus is not simply adding more features, but studying how recommendation performance changes when this information becomes increasingly sparse.
Research Objective
The study evaluates whether heterogeneous item-side information can improve sequential recommendation and whether adaptive gated fusion can make BERT4Rec more robust when some item-side information is unavailable.
The experiments compare:
Standard BERT4Rec using only item IDs.
BERT4Rec with heterogeneous item-side features.
BERT4Rec using conventional feature fusion.
BERT4Rec using adaptive gated fusion.
These models are evaluated under different levels of item-side feature availability.
