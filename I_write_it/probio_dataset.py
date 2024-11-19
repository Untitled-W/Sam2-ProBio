'''
the root of the dataset is '/home/minqi/Desktop/ProBio/ProBio_final'
the dataset is organized as follows:
ProBio_final
    |---train
        |---video1_folder
            |---gt
                |---000.png
                |---001.png
                |---...
            |---images
                |---000.jpg
                |---001.jpg
                |---...
            | ---label.json
        |---video2_folder
        |---...
    |---valid

The label.json file is a list with dictionary as items with the following structure:

[
    {
        "path": "string",
        "items": [
            {
                "segmentation": [
                    [float, float, ...]
                ],
                "object": "string",
                "object_id": "string",
                "solution": "string",
                "occlusion": "string"
            },
            ...
        ]
    },
    ...
]

The segmentation is a list of float numbers representing the polygon of the object.
The object is a string with the name of the object.
The object_id is a string with the id of the object.
The solution is a string with the solution of the object.
The occlusion is a string with the occlusion of the object.

Write me a class ProBioDataset that inherits from torch.utils.data.Dataset and implements the following methods:
    - __init__(self, root: str, split: str, transform: Optional[Callable] = None)
    - __len__(self) -> int
    - __getitem__(self, idx: int) -> Dict[str, Any]
    

The __init__ method should receive the root of the dataset and the split (either 'train' or 'valid') and an optional transform function.
The __len__ method should return the number of items in the dataset.
The __getitem__ method should return a dictionary with the following keys:
    - 'image': a PIL image
    - 'segmentation': a list of float numbers representing the polygon of the object
    - 'object': a string with the name of the object
    - 'object_id': a string with the id of the object
    - 'solution': a string with the solution of the object
    - 'occlusion': a string with the occlusion of the object
'''