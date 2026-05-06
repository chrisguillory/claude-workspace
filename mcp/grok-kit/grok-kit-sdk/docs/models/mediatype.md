# MediaType

Media type tags on a conversation. Observed values; xAI may add more.

## Example Usage

```python
from grok_kit_sdk.models import MediaType

# Open enum: unrecognized values are captured as UnrecognizedStr
value: MediaType = "audio"
```


## Values

This is an open enum. Unrecognized values will not fail type checks.

- `"audio"`
- `"text"`
- `"image"`
- `"video"`
