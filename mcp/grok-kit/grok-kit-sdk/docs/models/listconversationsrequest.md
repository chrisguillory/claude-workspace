# ListConversationsRequest


## Fields

| Field                                           | Type                                            | Required                                        | Description                                     |
| ----------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- |
| `page_size`                                     | *Optional[int]*                                 | :heavy_minus_sign:                              | Server caps observed at 60.                     |
| `page_token`                                    | *Optional[str]*                                 | :heavy_minus_sign:                              | Cursor from a prior response's `nextPageToken`. |
| `filter_is_starred`                             | *Optional[bool]*                                | :heavy_minus_sign:                              | Restrict to starred conversations only.         |