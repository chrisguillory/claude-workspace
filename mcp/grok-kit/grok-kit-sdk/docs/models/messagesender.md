# MessageSender

Originator of a message.

## Example Usage

```python
from grok_kit_sdk.models import MessageSender

# Open enum: unrecognized values are captured as UnrecognizedStr
value: MessageSender = "human"
```


## Values

This is an open enum. Unrecognized values will not fail type checks.

- `"human"`
- `"assistant"`
