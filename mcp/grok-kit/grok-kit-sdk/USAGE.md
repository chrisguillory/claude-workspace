<!-- Start SDK Example Usage [usage] -->
```python
# Synchronous Example
from grok_kit_sdk import GrokKit


with GrokKit(
    cookie_header="<YOUR_API_KEY_HERE>",
) as grok_kit:

    res = grok_kit.conversations.list_conversations(page_size=60)

    # Handle response
    print(res)
```

</br>

The same SDK client can also be used to make asynchronous requests by importing asyncio.

```python
# Asynchronous Example
import asyncio
from grok_kit_sdk import GrokKit

async def main():

    async with GrokKit(
        cookie_header="<YOUR_API_KEY_HERE>",
    ) as grok_kit:

        res = await grok_kit.conversations.list_conversations_async(page_size=60)

        # Handle response
        print(res)

asyncio.run(main())
```
<!-- End SDK Example Usage [usage] -->