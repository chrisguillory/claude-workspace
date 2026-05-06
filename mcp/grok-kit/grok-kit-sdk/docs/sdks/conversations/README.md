# Conversations

## Overview

### Available Operations

* [list_conversations](#list_conversations) - List conversations
* [get_conversation](#get_conversation) - Get conversation metadata
* [list_share_links](#list_share_links) - List share links for a conversation

## list_conversations

Returns a page of conversation summaries, ordered by modifyTime descending.
Pagination via `pageToken`; `nextPageToken` in the response is the cursor for
the next page (empty string when last page).


### Example Usage

<!-- UsageSnippet language="python" operationID="listConversations" method="get" path="/rest/app-chat/conversations" -->
```python
from grok_kit_sdk import GrokKit


with GrokKit(
    cookie_header="<YOUR_API_KEY_HERE>",
) as grok_kit:

    res = grok_kit.conversations.list_conversations(page_size=60)

    # Handle response
    print(res)

```

### Parameters

| Parameter                                                           | Type                                                                | Required                                                            | Description                                                         |
| ------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `page_size`                                                         | *Optional[int]*                                                     | :heavy_minus_sign:                                                  | Server caps observed at 60.                                         |
| `page_token`                                                        | *Optional[str]*                                                     | :heavy_minus_sign:                                                  | Cursor from a prior response's `nextPageToken`.                     |
| `filter_is_starred`                                                 | *Optional[bool]*                                                    | :heavy_minus_sign:                                                  | Restrict to starred conversations only.                             |
| `retries`                                                           | [Optional[utils.RetryConfig]](../../models/utils/retryconfig.md)    | :heavy_minus_sign:                                                  | Configuration to override the default retry behavior of the client. |

### Response

**[models.ConversationListResponse](../../models/conversationlistresponse.md)**

### Errors

| Error Type                 | Status Code                | Content Type               |
| -------------------------- | -------------------------- | -------------------------- |
| errors.GrokKitDefaultError | 4XX, 5XX                   | \*/\*                      |

## get_conversation

Returns metadata for a single conversation. Note `_v2` suffix.

### Example Usage

<!-- UsageSnippet language="python" operationID="getConversation" method="get" path="/rest/app-chat/conversations_v2/{conversationId}" -->
```python
from grok_kit_sdk import GrokKit


with GrokKit(
    cookie_header="<YOUR_API_KEY_HERE>",
) as grok_kit:

    res = grok_kit.conversations.get_conversation(conversation_id="d30826bd-a574-4f2f-a65e-a43a766ba3cd", include_workspaces=True, include_task_result=True)

    # Handle response
    print(res)

```

### Parameters

| Parameter                                                           | Type                                                                | Required                                                            | Description                                                         |
| ------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `conversation_id`                                                   | *str*                                                               | :heavy_check_mark:                                                  | N/A                                                                 |
| `include_workspaces`                                                | *Optional[bool]*                                                    | :heavy_minus_sign:                                                  | N/A                                                                 |
| `include_task_result`                                               | *Optional[bool]*                                                    | :heavy_minus_sign:                                                  | N/A                                                                 |
| `retries`                                                           | [Optional[utils.RetryConfig]](../../models/utils/retryconfig.md)    | :heavy_minus_sign:                                                  | Configuration to override the default retry behavior of the client. |

### Response

**[models.ConversationDetailResponse](../../models/conversationdetailresponse.md)**

### Errors

| Error Type                 | Status Code                | Content Type               |
| -------------------------- | -------------------------- | -------------------------- |
| errors.GrokKitDefaultError | 4XX, 5XX                   | \*/\*                      |

## list_share_links

List share links for a conversation

### Example Usage

<!-- UsageSnippet language="python" operationID="listShareLinks" method="get" path="/rest/app-chat/share_links" -->
```python
from grok_kit_sdk import GrokKit


with GrokKit(
    cookie_header="<YOUR_API_KEY_HERE>",
) as grok_kit:

    res = grok_kit.conversations.list_share_links(conversation_id="4f822020-28b1-42f7-86ec-1d1972e87bec", page_size=100)

    # Handle response
    print(res)

```

### Parameters

| Parameter                                                           | Type                                                                | Required                                                            | Description                                                         |
| ------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `conversation_id`                                                   | *str*                                                               | :heavy_check_mark:                                                  | N/A                                                                 |
| `page_size`                                                         | *Optional[int]*                                                     | :heavy_minus_sign:                                                  | N/A                                                                 |
| `retries`                                                           | [Optional[utils.RetryConfig]](../../models/utils/retryconfig.md)    | :heavy_minus_sign:                                                  | Configuration to override the default retry behavior of the client. |

### Response

**[models.ShareLinksResponse](../../models/sharelinksresponse.md)**

### Errors

| Error Type                 | Status Code                | Content Type               |
| -------------------------- | -------------------------- | -------------------------- |
| errors.GrokKitDefaultError | 4XX, 5XX                   | \*/\*                      |