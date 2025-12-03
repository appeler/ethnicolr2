# Florida Models

*This section is under development. For now, see the [API Reference](../api-reference/models.md) for technical details.*

## Coming Soon

- Detailed Florida model usage guide
- Comparison with census models
- Best practices for different use cases
- Performance optimization tips

## Quick Reference

```python
from ethnicolr2 import pred_fl_last_name, pred_fl_full_name

# Last name only
result = pred_fl_last_name(df, lname_col='last_name')

# Full name (highest accuracy)
result = pred_fl_full_name(df, lname_col='last_name', fname_col='first_name')
```
