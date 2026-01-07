# Kitchen Brigade Prompt Templates

This directory contains reusable prompt templates for Kitchen Brigade protocols.

## Template Variables

Templates support the following variable substitutions:

- `{{role}}` - Current LLM's role (e.g., "architect", "critic")
- `{{inputs}}` - JSON of protocol inputs
- `{{documents}}` - Content of input documents (if provided)
- `{{previous_rounds}}` - JSON of all previous round outputs

## Template Files

### Architecture Reconciliation
- `round1_analysis.txt` - Initial parallel analysis
- `round2_conflicts.txt` - Conflict identification (synthesis)
- `round3_resolutions.txt` - Resolution proposals (parallel)
- `round4_consensus.txt` - Final consensus (synthesis)

### WBS Generation
- `wbs_round1_phases.txt` - Phase breakdown
- `wbs_round2_tasks.txt` - Task decomposition
- `wbs_round3_validation.txt` - Validation
- `wbs_round4_finalize.txt` - Final WBS

## Adding New Templates

1. Create new protocol JSON in `config/protocols/`
2. Create corresponding prompt templates here
3. Reference template names in protocol JSON
4. Use `{{variables}}` for dynamic content
