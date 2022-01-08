# LVAD
## Data
filename
01_0014_alphapose_tracked_person.json
{scene_id}_{clip_id}_{method}.json
```json
{
    "type": "object",
    "properties": {
        "3": {
            "type": "object",
            "properties": {
                "014": {
                    "type": "object",
                    "properties": {
                        "keypoints": {
                            "type": "array",
                            "items": [
                                {
                                    "type": "number"
                                }
                            ]
                        },
                        "scores": {
                            "type": "number"
                        }
                    },
                    "required": [
                        "keypoints",
                        "scores"
                    ]
                }
            }
        }
    }
}
```


