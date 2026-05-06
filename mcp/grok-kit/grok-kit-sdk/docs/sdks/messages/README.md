# Messages

## Overview

### Available Operations

* [get_response_tree](#get_response_tree) - Get conversation message tree
* [load_messages](#load_messages) - Load full message bodies for given responseIds

## get_response_tree

Returns the response-tree for a conversation: a flat array of nodes, each with
responseId + sender + parentResponseId. Reconstruct the conversation thread by
following parent links from leaves up.


### Example Usage

<!-- UsageSnippet language="python" operationID="getResponseTree" method="get" path="/rest/app-chat/conversations/{conversationId}/response-node" -->
```python
from grok_kit_sdk import GrokKit


with GrokKit(
    cookie_header="<YOUR_API_KEY_HERE>",
) as grok_kit:

    res = grok_kit.messages.get_response_tree(conversation_id="b80bd10d-200c-4ba6-a6b6-2dd13a18b548", include_threads=True)

    # Handle response
    print(res)

```

### Parameters

| Parameter                                                           | Type                                                                | Required                                                            | Description                                                         |
| ------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `conversation_id`                                                   | *str*                                                               | :heavy_check_mark:                                                  | N/A                                                                 |
| `include_threads`                                                   | *Optional[bool]*                                                    | :heavy_minus_sign:                                                  | N/A                                                                 |
| `retries`                                                           | [Optional[utils.RetryConfig]](../../models/utils/retryconfig.md)    | :heavy_minus_sign:                                                  | Configuration to override the default retry behavior of the client. |

### Response

**[models.ResponseTreeResponse](../../models/responsetreeresponse.md)**

### Errors

| Error Type                 | Status Code                | Content Type               |
| -------------------------- | -------------------------- | -------------------------- |
| errors.GrokKitDefaultError | 4XX, 5XX                   | \*/\*                      |

## load_messages

Returns the full payload (text + metadata) for the supplied responseIds.
Server cap on responseIds per call has not been empirically determined;
observed working with up to 30 in a single request.


### Example Usage

<!-- UsageSnippet language="python" operationID="loadMessages" method="post" path="/rest/app-chat/conversations/{conversationId}/load-responses" -->
```python
from grok_kit_sdk import GrokKit


with GrokKit(
    cookie_header="<YOUR_API_KEY_HERE>",
) as grok_kit:

    res = grok_kit.messages.load_messages(conversation_id="313d13f6-ba44-422b-8294-0adaf4fa7757", response_ids=[
        "5ed3eb56-95f6-4332-9beb-e129f7e07ab0",
        "45579eb3-868a-404e-b53f-3a5390ee1912",
        "2ee7b7a2-846e-4fe4-9ef5-7a4acd6e2a39",
    ])

    # Handle response
    print(res)

```

### Parameters

| Parameter                                                           | Type                                                                | Required                                                            | Description                                                         |
| ------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `conversation_id`                                                   | *str*                                                               | :heavy_check_mark:                                                  | N/A                                                                 |
| `response_ids`                                                      | List[*str*]                                                         | :heavy_check_mark:                                                  | N/A                                                                 |
| `retries`                                                           | [Optional[utils.RetryConfig]](../../models/utils/retryconfig.md)    | :heavy_minus_sign:                                                  | Configuration to override the default retry behavior of the client. |

### Response

**[models.LoadResponsesResponse](../../models/loadresponsesresponse.md)**

### Errors

| Error Type                 | Status Code                | Content Type               |
| -------------------------- | -------------------------- | -------------------------- |
| errors.GrokKitDefaultError | 4XX, 5XX                   | \*/\*                      |