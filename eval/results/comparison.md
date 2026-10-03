| Run | Method | Recall@1 | Recall@5 | MRR | Relational MRR | Relational Hit@5 | Abstain P | Abstain R | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| latest | Lexical | 0.594 | 0.811 | 0.731 | 0.464 | 0.733 | 0.294 | 0.556 | 4.7 |
| latest | Vector | 0.490 | 0.760 | 0.653 | 0.276 | 0.400 | 0.269 | 0.389 | 3.9 |
| latest | Hybrid | 0.581 | 0.902 | 0.751 | 0.349 | 0.800 | 0.462 | 0.333 | 7.2 |
| latest | Hybrid + Entity | 0.581 | 0.927 | 0.751 | 0.357 | 0.933 | 0.688 | 0.611 | 12.5 |
| memory_backend | Lexical | 0.581 | 0.811 | 0.724 | 0.451 | 0.733 | 0.286 | 0.556 | 0.4 |
| memory_backend | Vector | 0.490 | 0.760 | 0.653 | 0.276 | 0.400 | 0.269 | 0.389 | 2.4 |
| memory_backend | Hybrid | 0.594 | 0.896 | 0.756 | 0.347 | 0.800 | 0.462 | 0.333 | 2.5 |
| memory_backend | Hybrid + Entity | 0.594 | 0.927 | 0.756 | 0.350 | 0.933 | 0.688 | 0.611 | 7.7 |
| ablation_no_graph_expansion | Lexical | 0.594 | 0.811 | 0.731 | 0.464 | 0.733 | 0.294 | 0.556 | 4.8 |
| ablation_no_graph_expansion | Vector | 0.490 | 0.760 | 0.653 | 0.276 | 0.400 | 0.269 | 0.389 | 3.9 |
| ablation_no_graph_expansion | Hybrid | 0.581 | 0.902 | 0.751 | 0.349 | 0.800 | 0.462 | 0.333 | 7.8 |
| ablation_no_graph_expansion | Hybrid + Entity | 0.594 | 0.872 | 0.746 | 0.295 | 0.600 | 0.688 | 0.611 | 11.9 |
| ablation_no_header_context | Lexical | 0.594 | 0.811 | 0.731 | 0.464 | 0.733 | 0.294 | 0.556 | 4.8 |
| ablation_no_header_context | Vector | 0.490 | 0.760 | 0.653 | 0.276 | 0.400 | 0.269 | 0.389 | 4.1 |
| ablation_no_header_context | Hybrid | 0.581 | 0.902 | 0.751 | 0.349 | 0.800 | 0.462 | 0.333 | 8.7 |
| ablation_no_header_context | Hybrid + Entity | 0.581 | 0.890 | 0.744 | 0.319 | 0.733 | 0.688 | 0.611 | 16.3 |
| ablation_no_header_no_hints | Lexical | 0.594 | 0.811 | 0.731 | 0.464 | 0.733 | 0.294 | 0.556 | 4.1 |
| ablation_no_header_no_hints | Vector | 0.490 | 0.760 | 0.653 | 0.276 | 0.400 | 0.269 | 0.389 | 4.6 |
| ablation_no_header_no_hints | Hybrid | 0.581 | 0.902 | 0.751 | 0.349 | 0.800 | 0.462 | 0.333 | 7.5 |
| ablation_no_header_no_hints | Hybrid + Entity | 0.581 | 0.884 | 0.743 | 0.312 | 0.667 | 0.688 | 0.611 | 12.7 |
| ablation_no_type_hints | Lexical | 0.594 | 0.811 | 0.731 | 0.464 | 0.733 | 0.294 | 0.556 | 4.3 |
| ablation_no_type_hints | Vector | 0.490 | 0.760 | 0.653 | 0.276 | 0.400 | 0.269 | 0.389 | 4.1 |
| ablation_no_type_hints | Hybrid | 0.581 | 0.902 | 0.751 | 0.349 | 0.800 | 0.462 | 0.333 | 7.4 |
| ablation_no_type_hints | Hybrid + Entity | 0.581 | 0.896 | 0.746 | 0.327 | 0.733 | 0.688 | 0.611 | 13.5 |
| ablation_rrf_fusion | Lexical | 0.594 | 0.811 | 0.731 | 0.464 | 0.733 | 0.314 | 0.611 | 4.3 |
| ablation_rrf_fusion | Vector | 0.490 | 0.760 | 0.653 | 0.276 | 0.400 | 0.280 | 0.389 | 4.0 |
| ablation_rrf_fusion | Hybrid | 0.496 | 0.880 | 0.684 | 0.312 | 0.733 | 0.500 | 0.389 | 7.3 |
| ablation_rrf_fusion | Hybrid + Entity | 0.496 | 0.884 | 0.685 | 0.332 | 0.733 | 0.647 | 0.611 | 13.0 |
| ablation_ts_rank | Lexical | 0.520 | 0.768 | 0.669 | 0.383 | 0.667 | 0.343 | 0.667 | 3.7 |
| ablation_ts_rank | Vector | 0.490 | 0.760 | 0.653 | 0.276 | 0.400 | 0.269 | 0.389 | 5.2 |
| ablation_ts_rank | Hybrid | 0.575 | 0.866 | 0.736 | 0.404 | 0.733 | 0.467 | 0.389 | 6.2 |
| ablation_ts_rank | Hybrid + Entity | 0.575 | 0.908 | 0.740 | 0.421 | 0.933 | 0.611 | 0.611 | 11.7 |
