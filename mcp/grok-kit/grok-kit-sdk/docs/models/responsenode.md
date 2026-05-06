# ResponseNode


## Fields

| Field                                                         | Type                                                          | Required                                                      | Description                                                   |
| ------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------- |
| `response_id`                                                 | *str*                                                         | :heavy_check_mark:                                            | N/A                                                           |
| `sender`                                                      | [models.MessageSender](../models/messagesender.md)            | :heavy_check_mark:                                            | Originator of a message.                                      |
| `parent_response_id`                                          | *Optional[str]*                                               | :heavy_minus_sign:                                            | Parent message in the conversation tree; null/absent on root. |