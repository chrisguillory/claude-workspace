# ConversationListResponse


## Fields

| Field                                                                | Type                                                                 | Required                                                             | Description                                                          |
| -------------------------------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `conversations`                                                      | List[[models.ConversationSummary](../models/conversationsummary.md)] | :heavy_check_mark:                                                   | N/A                                                                  |
| `next_page_token`                                                    | *Optional[str]*                                                      | :heavy_minus_sign:                                                   | Page cursor; empty string when last page.                            |
| `text_search_matches`                                                | List[Dict[str, *Any*]]                                               | :heavy_minus_sign:                                                   | Text-search match metadata; observed empty when no query.            |