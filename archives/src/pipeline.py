class Pipeline:
    """Base class for image processing pipeline"""

    def __init__(self, path_to_image: str, is_ref: bool):
        self.path = path_to_image
        self.state = "raw"
        self.ref = is_ref

    def __str__(self) -> str:
        return "{0}: [state = {1}] [ref = {2}]".format(self.path, self.state, self.ref)
    
    def __repr__(self) -> str:
        return f"Pipeline({self.path}, {self.ref})"