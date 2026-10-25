# CineMatch AI - evaluation report

Catalogue size: 4803 movies; TF-IDF matrix 4803 x 20000.

## Franchise-retrieval proxy
439 query movies in 174 title-stem groups. Relevant = other movies with the same title stem. This is a narrow proxy (sequel retrieval), not a measure of user satisfaction.

|   K |   Precision@K |   Recall@K |   Hit rate@K |
|----:|--------------:|-----------:|-------------:|
|   5 |         0.269 |      0.691 |        0.781 |
|  10 |         0.157 |      0.768 |        0.829 |
|  20 |         0.086 |      0.82  |        0.852 |

## Coverage and diversity (all movies as queries)
- Top-10 catalogue coverage: 97.7% (112 movies never recommended)
- Most-recommended movie appears in 157 lists
- Mean intra-list diversity (sample of 500): 0.840
- Mean top-1 similarity: 0.305; median top-10 similarity: 0.204
- Median vote count: catalogue 235 vs recommended 238

## Strong examples (franchise sibling at rank 1)
| title                                         |   n_relevant |   P@5 |   top1_score |
|:----------------------------------------------|-------------:|------:|-------------:|
| Star Trek: First Contact (1996)               |           10 |     1 |        0.584 |
| Star Trek: Insurrection (1998)                |           10 |     1 |        0.584 |
| Star Trek V: The Final Frontier (1989)        |           10 |     1 |        0.527 |
| Star Trek VI: The Undiscovered Country (1991) |           10 |     1 |        0.527 |
| Star Trek: Nemesis (2002)                     |           10 |     1 |        0.513 |

## Weak examples (no franchise sibling in top 20)
| title                                       |   n_relevant |   top1_score |
|:--------------------------------------------|-------------:|-------------:|
| Halloween III: Season of the Witch (1982)   |            6 |        0.233 |
| Paranormal Activity: The Marked Ones (2014) |            4 |        0.419 |
| Scary Movie 5 (2013)                        |            3 |        0.279 |
| Batman (1966)                               |            2 |        0.318 |
| Insidious: Chapter 3 (2015)                 |            2 |        0.49  |
